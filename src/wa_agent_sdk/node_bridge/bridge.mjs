import fs from "node:fs";
import path from "node:path";
import pino from "pino";
import { WebSocketServer } from "ws";

const PORT = Number(process.env.WA_BRIDGE_PORT || 8765);
const TOKEN = process.env.WA_BRIDGE_TOKEN || "";
const AUTH_DIR = process.env.WA_AUTH_DIR || path.join(process.cwd(), ".wa_auth");
const LOG_LEVEL = process.env.WA_LOG_LEVEL || "error";

const logger = pino({ level: LOG_LEVEL });
const silent = pino({ level: "silent" });
fs.mkdirSync(AUTH_DIR, { recursive: true });

const MEDIA_CACHE_LIMIT = 200;
const mediaCache = new Map();

let wa = null;
let sock = null;
let client = null;
let restarting = false;

function send(obj) {
  if (client && client.readyState === 1) {
    try {
      client.send(JSON.stringify(obj));
    } catch {
      /* connection closing */
    }
  }
}

function result(ref, ok, data = {}) {
  send({ type: "result", ref, ok, ...data });
}

function emit(type, payload = {}) {
  send({ type, ...payload });
}

async function loadBaileys() {
  const mod = await import("@whiskeysockets/baileys");
  const m = mod.default ?? mod;
  const makeWASocket = m.default ?? m;
  return {
    makeWASocket,
    useMultiFileAuthState: m.useMultiFileAuthState,
    fetchLatestBaileysVersion: m.fetchLatestBaileysVersion,
    DisconnectReason: m.DisconnectReason ?? {},
    downloadMediaMessage: m.downloadMediaMessage,
  };
}

function unwrap(message) {
  let cur = message;
  for (let i = 0; i < 4; i++) {
    const nested =
      cur?.ephemeralMessage?.message ||
      cur?.viewOnceMessage?.message ||
      cur?.viewOnceMessageV2?.message ||
      cur?.documentWithCaptionMessage?.message;
    if (!nested) break;
    cur = nested;
  }
  return cur;
}

function normalize(raw) {
  const jid = raw?.key?.remoteJid || "";
  if (!jid || jid === "status@broadcast" || jid.endsWith("@newsletter")) return null;
  const inner = unwrap(raw.message);
  if (!inner) return null;

  let media_type = "other";
  let text = null;
  let caption = null;
  let mimetype = null;
  let filename = null;
  let has_media = false;

  if (inner.conversation != null) {
    media_type = "text";
    text = inner.conversation;
  } else if (inner.extendedTextMessage) {
    media_type = "text";
    text = inner.extendedTextMessage.text || null;
  } else if (inner.imageMessage) {
    media_type = "image";
    caption = inner.imageMessage.caption || null;
    mimetype = inner.imageMessage.mimetype || null;
    has_media = true;
  } else if (inner.videoMessage) {
    media_type = "video";
    caption = inner.videoMessage.caption || null;
    mimetype = inner.videoMessage.mimetype || null;
    has_media = true;
  } else if (inner.audioMessage) {
    media_type = "audio";
    mimetype = inner.audioMessage.mimetype || null;
    has_media = true;
  } else if (inner.stickerMessage) {
    media_type = "sticker";
    mimetype = inner.stickerMessage.mimetype || null;
    has_media = true;
  } else if (inner.documentMessage) {
    media_type = "document";
    mimetype = inner.documentMessage.mimetype || null;
    filename =
      inner.documentMessage.fileName || inner.documentMessage.title || "document";
    caption = inner.documentMessage.caption || null;
    has_media = true;
  } else if (inner.contactMessage) {
    media_type = "contact";
    text = `Contact: ${inner.contactMessage.displayName || ""}`;
  } else if (inner.locationMessage) {
    media_type = "location";
    text = `Location: ${inner.locationMessage.degreesLatitude},${inner.locationMessage.degreesLongitude}`;
  } else if (inner.liveLocationMessage) {
    media_type = "location";
    text = `Live location: ${inner.liveLocationMessage.degreesLatitude},${inner.liveLocationMessage.degreesLongitude}`;
  } else {
    return null;
  }

  const ctxInfo =
    inner.extendedTextMessage?.contextInfo ||
    inner.imageMessage?.contextInfo ||
    inner.videoMessage?.contextInfo ||
    inner.documentMessage?.contextInfo ||
    null;
  const mentionedJids = Array.isArray(ctxInfo?.mentionedJid) ? ctxInfo.mentionedJid : [];
  const quotedParticipant = ctxInfo?.participant || null;

  return {
    id: raw.key.id || "",
    jid,
    sender_jid: raw.key.participant || jid,
    push_name: raw.pushName || null,
    from_me: !!raw.key.fromMe,
    is_group: jid.endsWith("@g.us"),
    media_type,
    text: text || caption || null,
    caption,
    mimetype,
    filename,
    has_media,
    mentioned_jids: mentionedJids,
    quoted_participant: quotedParticipant,
    timestamp: Number(raw.messageTimestamp || 0) * 1000,
  };
}

function cacheMedia(raw, norm) {
  if (!norm.has_media || !norm.id) return;
  if (mediaCache.size >= MEDIA_CACHE_LIMIT) {
    mediaCache.delete(mediaCache.keys().next().value);
  }
  mediaCache.set(norm.id, { raw, mimetype: norm.mimetype });
}

const withTimeout = (promise, ms, label) =>
  Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
    ),
  ]);

async function startWhatsApp() {
  const { state, saveCreds } = await wa.useMultiFileAuthState(AUTH_DIR);
  let version;
  try {
    ({ version } = await withTimeout(wa.fetchLatestBaileysVersion(), 10000, "version fetch"));
  } catch (err) {
    logger.warn({ err: String(err?.message || err) }, "version fetch failed; using baked-in default");
    version = undefined;
  }

  sock = wa.makeWASocket({
    version,
    auth: state,
    logger: silent,
    printQRInTerminal: false,
    browser: ["Wa Agent SDK", "Chrome", "1.0.0"],
    markOnlineOnConnect: false,
    syncFullHistory: false,
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", (u) => {
    const { connection, lastDisconnect, qr } = u;
    if (qr) emit("qr", { qr });
    if (connection === "open") {
      emit("ready", { jid: sock.user?.id || null, name: sock.user?.name || null });
    }
    if (connection === "close") {
      const code = lastDisconnect?.error?.output?.statusCode;
      const loggedOut = code === wa.DisconnectReason.loggedOut;
      emit("disconnected", {
        code: Number(code || 0),
        logged_out: !!loggedOut,
        reason: String(lastDisconnect?.error?.message || ""),
      });
      sock = null;
      if (!loggedOut && !restarting) {
        setTimeout(() => {
          startWhatsApp().catch((err) => emit("fatal", { error: String(err?.stack || err) }));
        }, 3000);
      }
    }
  });

  sock.ev.on("messages.upsert", ({ messages }) => {
    for (const raw of messages) {
      const norm = normalize(raw);
      if (!norm) continue;
      if (norm.has_media) cacheMedia(raw, norm);
      emit("message", { payload: norm });
    }
  });
}

function buildMediaContent(p) {
  const buf = Buffer.from(p.data_b64 || "", "base64");
  switch (p.media_type) {
    case "image":
      return { image: buf, caption: p.caption || undefined, mimetype: p.mimetype || undefined };
    case "video":
      return { video: buf, caption: p.caption || undefined, mimetype: p.mimetype || undefined };
    case "audio":
      return { audio: buf, mimetype: p.mimetype || "audio/mpeg", ptt: !!p.ptt };
    case "sticker":
      return { sticker: buf };
    default:
      return {
        document: buf,
        mimetype: p.mimetype || "application/octet-stream",
        fileName: p.filename || "file",
        caption: p.caption || undefined,
      };
  }
}

async function handleCommand(data) {
  const { type, ref } = data;
  try {
    if (type === "ping") return result(ref, true, { pong: true });

    if (type === "send_text") {
      if (!sock) throw new Error("not_connected");
      const r = await sock.sendMessage(data.to, { text: data.text });
      return result(ref, true, { id: r?.key?.id || null });
    }

    if (type === "send_media") {
      if (!sock) throw new Error("not_connected");
      const r = await sock.sendMessage(data.to, buildMediaContent(data));
      return result(ref, true, { id: r?.key?.id || null });
    }

    if (type === "set_presence") {
      if (sock) await sock.sendPresenceUpdate(data.presence, data.jid || undefined);
      return result(ref, true);
    }

    if (type === "mark_read") {
      if (sock) {
        const key = {
          remoteJid: data.chat_jid,
          id: data.id,
          fromMe: false,
          participant: data.sender_jid || undefined,
        };
        await sock.readMessages([key]);
      }
      return result(ref, true);
    }

    if (type === "download_media") {
      const cached = mediaCache.get(data.id);
      if (!cached) return result(ref, false, { error: "media_not_found" });
      const buf = await wa.downloadMediaMessage(cached.raw, "buffer", {}, {
        logger: silent,
        reuploadRequest: sock?.reuploadRequest,
      });
      return result(ref, true, {
        data_b64: Buffer.from(buf).toString("base64"),
        mimetype: cached.mimetype || null,
      });
    }

    if (type === "logout") {
      restarting = true;
      try {
        await sock?.logout();
      } catch {
        /* already unlinked */
      }
      try {
        fs.rmSync(AUTH_DIR, { recursive: true, force: true });
      } catch {
        /* best effort */
      }
      result(ref, true);
      setTimeout(() => process.exit(0), 250);
      return;
    }

    result(ref, false, { error: `unknown_command:${type}` });
  } catch (err) {
    result(ref, false, { error: String(err?.message || err) });
  }
}

const wss = new WebSocketServer({ host: "127.0.0.1", port: PORT });

wss.on("listening", () => {
  logger.info({ port: PORT }, "bridge listening");
});


wss.on("connection", (ws, req) => {
  const url = new URL(req.url || "/", "http://127.0.0.1");
  if (TOKEN && url.searchParams.get("token") !== TOKEN) {
    ws.close(4001, "unauthorized");
    return;
  }
  if (client && client.readyState === 1) {
    ws.close(4002, "already-connected");
    return;
  }
  client = ws;

  ws.on("message", (buf) => {
    let data;
    try {
      data = JSON.parse(buf.toString("utf8"));
    } catch {
      return;
    }
    Promise.resolve(handleCommand(data)).catch((err) => {
      if (data && data.ref != null) result(data.ref, false, { error: String(err) });
    });
  });
  ws.on("close", () => {
    if (client === ws) client = null;
  });
  ws.on("error", () => {});

  emit("hello", { pid: process.pid, bridge: "1.0.0" });
});

(async () => {
  try {
    wa = await loadBaileys();
  } catch (err) {
    emit("fatal", { error: `failed_to_load_baileys: ${String(err?.message || err)}` });
    setTimeout(() => process.exit(1), 300);
    return;
  }
  try {
    await startWhatsApp();
  } catch (err) {
    emit("fatal", { error: String(err?.stack || err) });
  }
})();

process.on("uncaughtException", (err) => {
  logger.error({ err: String(err?.stack || err) }, "uncaughtException");
});
process.on("unhandledRejection", (err) => {
  logger.error({ err: String(err) }, "unhandledRejection");
});
