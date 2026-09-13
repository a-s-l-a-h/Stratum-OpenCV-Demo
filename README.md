\---

# 🚀 Stratum Android Ongoing Examples – Chaquopy + Python 3.10

<p align="center">
  <b>Android + Python integration using Stratum \& Chaquopy</b><br/>
  Clean examples • Ready to run • Beginner friendly
</p>



\---

# This demo need python 3.10 on your pc for Chaquopy 

## 📌 Example Branch

This example is based on the **develop branch** so it can crash:

👉 https://github.com/a-s-l-a-h/stratum



## 📦 Included Examples



\---

## ⚙️ Requirements

<div align="center">

|Tool|Version|
|-|-|
|Android Studio|Latest|
|Python|**3.10**|
|Java|**17**|

</div>

> ⚠️ These examples are strictly built for \*\*Stratum v0.2 + Python 3.10\*\*

\---

## Python 3.10 + Chaquopy Setup

### 🔹 Step 1: Install Python 3.10

Install Python and note the path:

```text
C:/Python310/python.exe
```

\---

### 🔹 Step 2: Configure `build.gradle` (Module: app)

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

\---

## ❓ Why Python 3.10?

* The included `.whl` is built specifically for **Python 3.10**
* Other versions may cause build or runtime issues
* Chaquopy depends on your local Python version

\---

## 📁 Project Notes

### ✅ Stratum `.whl`

* 📍 Location: `app/libs/`
* Already included in all examples
* Built for \*\*Stratum \*\*

\---

## ▶️ Run the Examples

```bash
1. Open project in Android Studio
2. Sync Gradle
3. Run on emulator/device
```

\---

### 🔹 What is this?

* Defines **Stratum pipeline stages**
* Used internally by the Stratum build system
* Included for reference in each example

\---

## 📌 Summary

* ✅ Built for **Stratum v0.2**
* Requires **Python 3.10**
* 📦 `.whl` included → no extra setup
* ⚙️ Ready-to-run Android examples
* 🧩 Includes Stratum pipeline configs

\---

## 💡 Tips

* Keep Python path correct in `build.gradle`
* Don’t upgrade Python unless `.whl` supports it
* If build fails → recheck Chaquopy config

\---

<p align="center">
  💡 Are you using Stratum? Let us know what you think!<br/>
  Your feedback helps shape future updates and examples.
</p>

\---

