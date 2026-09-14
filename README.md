# 🚀 Stratum Android OpenCV Examples – Chaquopy + Python 3.10

<p align="center">
  <b>Android + Python integration using Stratum & Chaquopy</b><br/>
  Clean examples • Ready to run • Beginner friendly
</p>

---


<p align="center">
  <b>Android + Python integration using Stratum & Chaquopy</b><br/>
  Clean examples • Ready to run • Beginner friendly
</p>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <video src="https://github.com/user-attachments/assets/b8a2ae54-2b34-4636-9b95-27ed38a2e503" autoplay loop muted playsinline width="320"></video>
      </td>
    </tr>
  </table>
</div>

---



> [!CAUTION]
> ### ⚠️ CRITICAL COMPATIBILITY REQUIREMENTS
> - **Python 3.10 on PC:** You **MUST** have **Python 3.10** installed on your host PC for Chaquopy to build the project.
> - **Architecture Support:** This demo app runs **ONLY on `arm64` mobile devices** (`arm64-v8a`).
> - **Stratum Wheel:** This repository **only includes the Stratum `.whl` for `arm64-v8a`**, which is located in the `app/libs/` folder.

---

## 📌 Example Branch

This example is based on the **develop branch** so it can crash:

👉 **Repository:** [https://github.com/a-s-l-a-h/stratum](https://github.com/a-s-l-a-h/stratum)

---

## 📦 Included Examples

*(Examples and samples will be populated here)*

---

## ⚙️ Requirements

<div align="center">

| Tool | Supported Version |
| :--- | :--- |
| **Android Studio** | Latest |
| **Python** | **3.10** |
| **Java** | **17** |

</div>

> ⚠️ **Notice:** These examples are strictly built for **Stratum v0.2 + Python 3.10**.

---

## 🛠️ Setup Guide

### 🔹 Step 1: Install Python 3.10
Install Python 3.10 on your host PC and note its exact executable path:
```text
C:/Python310/python.exe
```

---

### 🔹 Step 2: Configure `build.gradle` (Module: app)

Ensure your build script points to your Python 3.10 installation and local library path:

```gradle
chaquopy {
    defaultConfig {
        version = "3.10"

        // Path to your local Python 3.10 installation
        buildPython("C:/Python310/python.exe")

        pip {
            // Use local .whl from libs folder
            options("--find-links", "${rootDir}/libs")

            // Install Stratum from local wheel
            install("stratum==0.9.0")
        }
    }
}
```

---

### 🔹 Step 3: Download Model Asset

You must manually download the NanoDet ONNX model file and place it inside your project's assets directory:

1. **Download:** [object_detection_nanodet_2022nov.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx)
2. **Target Destination:**
   ```text
   app/src/main/assets/object_detection_nanodet_2022nov.onnx
   ```

---

## ❓ Why Python 3.10?

* The included `.whl` is built specifically for **Python 3.10**.
* Other Python versions may cause build or runtime crashes.
* Chaquopy relies directly on your local system's Python installation during Gradle sync/build.

---

## 📁 Project Notes

### ✅ Stratum `.whl`
* 📍 **Location:** `app/libs/`
* Already included in all examples.
* Built for **Stratum**.

### 🔹 What is this?
* Defines **Stratum pipeline stages**.
* Used internally by the Stratum build system.
* Included for reference in each example.

---

## ▶️ Run the Examples

```bash
1. Open the project in Android Studio
2. Sync Gradle
3. Run on your physical arm64 mobile device / arm64 emulator
```

---

## 📌 Summary

* ✅ Built for **Stratum v0.9**
* 🐍 Requires **Python 3.10**
* 📦 `.whl` included → no extra setup
* ⚙️ Ready-to-run Android examples
* 🧩 Includes Stratum pipeline configs

---

## 💡 Tips

* Keep the Python path accurate in your `build.gradle`.
* Do not upgrade your local Python version unless the `.whl` supports it.
* If the build fails → verify your Chaquopy configuration and local Python path.

---

<p align="center">
  💡 <b>Are you using Stratum? Let us know what you think!</b><br/>
  Your feedback helps shape future updates and examples.
</p>
```
