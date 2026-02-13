#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const nodemailer = require("nodemailer");
const { execFile } = require("child_process");

const TARGET_URL = "https://huggingface.co/deepseek-ai";
const KEYWORD = "DeepSeek-V4"; // make it specific to avoid false positives
const KEYWORD_REGEX = /DeepSeek[-\s]?V4/;
const AUTH_TOKEN = ""; // no auth
const INTERVAL_MS = 3 * 60 * 1000; // 3 minutes
const TIMEOUT_MS = 10 * 1000; // 10 seconds
const RETRY_MAX = 3;
const RETRY_DELAY_MS = 5 * 1000;
const COOLDOWN_MS = 2 * 60 * 60 * 1000; // 2 hours
const SEND_ERROR_EMAIL = false; // stop error emails
let lastAlertAt = 0;
let lastErrorAlertAt = 0;

const LOG_PATH = "/home/ubuntu/.openclaw/workspace/tools/monitor-deepseek-v4.log";

const MAIL = {
  host: "smtp.163.com",
  port: 465,
  secure: true,
  auth: {
    user: "tiejipiaoliu@163.com",
    pass: "JNpaxhanY4bXXdmx",
  },
  from: "tiejipiaoliu@163.com",
  to: "tiejipiaoliu@163.com",
};

function log(message, extra) {
  const line = `[${new Date().toISOString()}] ${message}` + (extra ? ` | ${extra}` : "");
  fs.appendFileSync(LOG_PATH, line + "\n", "utf8");
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchWithRetry(url) {
  let lastErr;
  for (let attempt = 1; attempt <= RETRY_MAX; attempt += 1) {
    try {
      const args = [
        "-sS",
        "--max-time",
        String(Math.ceil(TIMEOUT_MS / 1000)),
        "--http1.1",
        "--tlsv1.2",
        "-H",
        "User-Agent: openclaw-monitor/1.0",
      ];
      if (AUTH_TOKEN) {
        args.push("-H", `Authorization: Bearer ${AUTH_TOKEN}`);
      }
      args.push(url);

      const text = await new Promise((resolve, reject) => {
        execFile("curl", args, { maxBuffer: 5 * 1024 * 1024 }, (err, stdout, stderr) => {
          if (err) {
            reject(new Error(stderr || err.message));
            return;
          }
          resolve(stdout);
        });
      });

      return text;
    } catch (err) {
      lastErr = err;
      log(`Fetch attempt ${attempt} failed`, err?.message || String(err));
      if (attempt < RETRY_MAX) await sleep(RETRY_DELAY_MS);
    }
  }
  throw lastErr;
}

async function sendMail(subject, text) {
  const transporter = nodemailer.createTransport({
    host: MAIL.host,
    port: MAIL.port,
    secure: MAIL.secure,
    auth: MAIL.auth,
  });

  await transporter.sendMail({
    from: MAIL.from,
    to: MAIL.to,
    subject,
    text,
  });
}

function formatLocalTime(date = new Date()) {
  return date.toLocaleString("zh-CN", {
    timeZone: "Asia/Shanghai",
    hour12: false,
  });
}

async function checkOnce() {
  try {
    const html = await fetchWithRetry(TARGET_URL);
    const match = KEYWORD_REGEX.exec(html);
    const hasKeyword = Boolean(match);
    log("Check completed", hasKeyword ? "KEYWORD_FOUND" : "no keyword");

    if (hasKeyword) {
      const now = Date.now();
      if (now - lastAlertAt < COOLDOWN_MS) {
        log("Keyword found but in cooldown", `cooldownMs=${COOLDOWN_MS}`);
        return;
      }
      lastAlertAt = now;
      const subject = "[ALERT] DeepSeek-V4 detected";
      const sendTime = formatLocalTime();
      const body = `Keyword '${KEYWORD}' detected on ${TARGET_URL} at ${sendTime} (send time)`;
      await sendMail(subject, body);
      log("Alert email sent", `sendTime=${sendTime}`);
    }
  } catch (err) {
    const subject = "[ERROR] DeepSeek monitor failed";
    const sendTime = formatLocalTime();
    const body = `Monitor failed at ${sendTime} (send time)\nError: ${err?.message || String(err)}`;
    if (!SEND_ERROR_EMAIL) {
      log("Error email suppressed", err?.message || String(err));
      return;
    }
    try {
      const now = Date.now();
      if (now - lastErrorAlertAt < COOLDOWN_MS) {
        log("Error in cooldown", `cooldownMs=${COOLDOWN_MS}`);
        return;
      }
      lastErrorAlertAt = now;
      await sendMail(subject, body);
      log("Error email sent");
    } catch (mailErr) {
      log("Error email failed", mailErr?.message || String(mailErr));
    }
  }
}

async function main() {
  log("Monitor started", `interval=${INTERVAL_MS}ms`);
  await checkOnce();
  setInterval(checkOnce, INTERVAL_MS);
}

process.on("unhandledRejection", (err) => log("UnhandledRejection", err?.message || String(err)));
process.on("uncaughtException", (err) => log("UncaughtException", err?.message || String(err)));

main();
