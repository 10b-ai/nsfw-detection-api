# 🫣 NSFW Detection API

A simple and scalable **NSFW image detection API** built with **FastAPI**, **PyTorch**, and Hugging Face's [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection). Deploys easily to **Railway** with Docker.

---

## 🔍 Features

- ✅ Detects NSFW content (porn, hentai, sexy, neutral, etc.)
- 🧠 Returns prediction scores, top label, and `nsfw: true/false`
- 🚀 FastAPI backend with Hugging Face pipeline
- 🎛 Customizable NSFW threshold
- 🐳 Dockerized for easy deployment
- ☁️ Works great on Railway

---

## 🚀 Quickstart

### 📦 Install Locally

```bash
git clone https://github.com/JarJarMadeIt/nsfw-detection-api.git
cd nsfw-detection-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 🐳 Run with Docker

```bash
docker build -t nsfw-detection-api .
docker run -p 8080:8080 nsfw-detection-api
```

---

## 🌐 API

### `POST /detect`

Accepts either an uploaded image file or a remote image URL, plus an optional threshold.

#### 📥 Request

- `Content-Type`: `multipart/form-data`
- Field: `image` or `image_url` (provide exactly one)
- Optional Field: `threshold` (float, default: `0.7`)

#### Example with `curl`

```bash
curl -X POST "http://localhost:8080/detect" \
     -F "image=@/path/to/image.jpg" \
     -F "threshold=0.7"
```

#### Example with remote image URL

```bash
curl -X POST "http://localhost:8080/detect" \
     -F "image_url=https://example.com/image.jpg" \
     -F "threshold=0.7"
```

#### 📤 Response

```json
{
  "scores": {
    "normal": 0.95,
    "nsfw": 0.05
  },
  "top": {
    "label": "normal",
    "confidence": 0.95
  },
  "nsfw": false
}
```

---

## 🧠 How It Works

This API uses [`transformers`](https://huggingface.co/docs/transformers/index) to load the `Falconsai/nsfw_image_detection` model and classify images using a Hugging Face pipeline.

---

## ☁️ Deploy on Railway

### One-click deploy

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template/Tlmof_?referralCode=DEwPnF)

### Using Railway CLI

```bash
railway init
railway up
```

## 🛠 Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Torch](https://pytorch.org/)
- [Pillow](https://pillow.readthedocs.io/)
- [Docker](https://www.docker.com/)
- [Railway](https://railway.app/)

---

## 📁 Project Structure

```
nsfw-api/
├── app/
│   └── main.py
├── requirements.txt
├── Dockerfile
└── railway.json
```

---

## 📄 License

MIT – use it, fork it, improve it. Stay safe 🔒

---

Built with ❤️ to keep the web a little safer.
