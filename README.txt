
# Community Detection Research Project

## Overview
This project demonstrates **community detection in social networks** using multiple algorithms such as:
- Louvain Algorithm
- Label Propagation
- Girvan–Newman Algorithm
- GCN Label Propagation (Neural Network-based approach)

It includes full **visualization**, **metric evaluation**, and **data export** for research and analysis.

---

## 🧠 Requirements

### 1. Install Python
Download and install Python 3.10+ from: https://www.python.org/downloads/
Make sure to check **"Add Python to PATH"** during installation.

### 2. Install Visual Studio Code
Download from: https://code.visualstudio.com/

Install the **Python extension** by Microsoft from the VS Code Marketplace.

---

## ⚙️ Setup Steps

### Step 1: Create Project Folder
Create a folder, e.g. `C:\Users\<YourName>\Desktop\CommunityDetection`

Copy the following files into the folder:
- `louvain_community_bar_publish.py`
- `newman_community_bar_publish.py`

### Step 2: Create Virtual Environment
Open VS Code Terminal (`Ctrl + ~`) and run:

```bash
python -m venv venv
```

Activate the environment:

- **Windows PowerShell:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies

```bash
pip install networkx matplotlib numpy pandas psutil torch python-louvain scikit-learn
```

If you face issues installing `torch`, use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Running the Scripts

### Option 1: Run via VS Code
- Open the file in VS Code.
- Click on the **Run ▶️** button at the top-right corner.

### Option 2: Run via Terminal
```bash
python louvain_community_bar_publish.py
```
or
```bash
python newman_community_bar_publish.py
```

---

## 📊 Results

After running the scripts, a **`results/`** folder will be created automatically.

It contains:
- `community_sizes.png` → Bar chart of detected communities
- `community_network.png` → Graph visualization of community structure
- `performance_metrics.png` → Accuracy, modularity, scalability graph
- `communities.csv` → Community data table
- `summary.json` → Summary of metrics and outputs

---

## ⚡ Optional Arguments

You can provide custom parameters:

### Example 1: Use a specific algorithm
```bash
python louvain_community_bar_publish.py --algorithm gcn_labelprop
```

### Example 2: Provide an input edge list
```bash
python newman_community_bar_publish.py --input graph_edges.txt
```

---

## 📈 Algorithms Description

| Algorithm | Description |
|------------|-------------|
| **Louvain** | Modularity optimization-based fast detection algorithm |
| **Label Propagation** | Detects communities via label diffusion |
| **Girvan–Newman** | Detects communities via edge betweenness centrality |
| **GCN Label Propagation** | Combines neural graph learning with propagation |

---

## 🧾 Credits
**Author:** Harihar Yadav  
**Institution:** CGC Jhanjeri  
**Project:** Graph Algorithms for Community Detection in Social Networks  
**Year:** 2025

---
