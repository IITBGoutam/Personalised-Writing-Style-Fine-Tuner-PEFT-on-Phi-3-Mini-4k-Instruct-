
## 🧩 **Dataset Creation Process**

The goal was to create a **parallel dataset** where each example teaches the model how to rewrite a generic, neutral text into **your personal writing style**.

---

### **Step 1: Collect user-style text**

You started with your own writings — articles, notes, or short paragraphs — that represent your natural phrasing, tone, and rhythm.
Example raw snippet:

```
This along with other data security concerns have led many big publications and research institutes to draft policies.
```

---

### **Step 2: Clean and structure the text**

A preprocessing script removed very short or incomplete lines and split longer text into meaningful sentences.
Each sentence was wrapped in a tag like:

```
[STYLE_SAMPLE]
This along with other data security concerns have led many big publications and research institutes to draft policies.
```

This formed a clean list of your style sentences.

---

### **Step 3: Generate neutral / GPT-style rewrites**

For each of your style samples, ChatGPT (or any base model) was asked to **rephrase it neutrally**, removing stylistic features.
Example:

```
Your text (style): This along with other data security concerns have led many big publications and research institutes to draft policies.
GPT rephrase: Data security concerns have prompted several institutions to create new policies.
```

---

### **Step 4: Pair them into training examples**

Each (neutral → stylized) pair was then formatted into the **Phi-3-style instruction format**:

```
<|system|>
You are a writing assistant that rewrites text in the author's personal style.<|end|>

<|user|>
Data security concerns have prompted several institutions to create new policies.<|end|>

<|assistant|>
This along with other data security concerns have led many big publications and research institutes to draft policies.<|end|>
```

Each of these examples teaches the model *how to convert GPT-style writing into your style.*

---

### **Step 5: Final dataset**

* Around **75 examples** were created.
* Each example contained:

  * A **neutral sentence** (input)
  * A **user-style sentence** (target)
* Saved in  `.txt` format for fine-tuning.



