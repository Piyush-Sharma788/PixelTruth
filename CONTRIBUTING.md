# Contributing to PixelTruth

Thank you for your interest in contributing to **PixelTruth**!

PixelTruth is an AI-powered project that helps detect whether an image is real or AI-generated. We welcome contributions from everyone, including beginners, open-source contributors, and GSSoC'26 participants.

---

## 📌 About the Project

PixelTruth focuses on image authenticity detection using AI/ML techniques.

Contributions can include:

- UI improvements
- Bug fixes
- Documentation updates
- Model improvements
- Feature additions
- Explainability improvements
- Testing and performance improvements

---

## 🤝 How to Contribute

Follow these steps to contribute to the project:

### 1. Fork the Repository

Click on the **Fork** button at the top-right corner of the repository page.

### 2. Clone Your Forked Repository

```bash
git clone https://github.com/your-username/PixelTruth.git
```

### 3. Move into the Project Directory

```bash
cd PixelTruth
```

### 4. Create a New Branch

Create a branch with a meaningful name:

```bash
git checkout -b feature/your-feature-name
```

Examples:

```bash
git checkout -b feature/batch-image-analysis
git checkout -b feature/grad-cam-visualization
git checkout -b fix/upload-error
```

### 5. Install Dependencies

Python **3.11** is required (TensorFlow 2.16).

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

The trained model (`deepfake_detection_model.h5`) is included in the repository, so no download is needed.

### 6. Run the Project Locally

```bash
streamlit run app.py
```

### 7. Make Your Changes

Work only on the issue assigned to you.

Keep your code clean, readable, and relevant to the issue.

### 8. Commit Your Changes

```bash
git add .
git commit -m "Add your feature"
```

### 9. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 10. Open a Pull Request

Open a Pull Request from your forked repository to the main PixelTruth repository.

---

## ✅ Pull Request Guidelines

Before creating a Pull Request, make sure:

- Your code runs without errors
- Your changes are related to the assigned issue
- You have tested your changes locally
- Your PR title is clear and meaningful
- Your PR description explains what you changed
- You have added screenshots or screen recordings for UI changes
- You have not made unnecessary changes in unrelated files

Use this format in your PR description:

```md
## Description
Briefly describe the changes made.

## Related Issue
Closes #issue_number

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] UI improvement
- [ ] Model improvement

## Screenshots / Demo
Add screenshots or screen recording if applicable.

## Checklist
- [ ] I have tested my changes locally
- [ ] My code follows the project guidelines
- [ ] I have only worked on the assigned issue
```

---

## 🐛 Reporting Bugs

If you find a bug, please create an issue with the following details:

```md
## Bug Description
Describe the bug clearly.

## Steps to Reproduce
1. Go to...
2. Click on...
3. See error...

## Expected Behavior
Explain what should happen.

## Screenshots
Add screenshots if applicable.

## Environment
- OS:
- Browser:
- Python version:
```

---

## 💡 Suggesting Features

If you want to suggest a feature, create an issue with this format:

```md
## Feature Description
Explain the feature clearly.

## Problem It Solves
Describe why this feature is needed.

## Proposed Solution
Explain how it can be implemented.

## Benefits
Mention how it improves the project.
```

---

## 🚀 Good Contribution Ideas

Here are some useful contribution ideas for PixelTruth:

- Improve Streamlit UI
- Add batch image analysis
- Add CSV export for prediction results
- Add Grad-CAM visualization
- Improve model explainability
- Improve image preprocessing
- Add better error handling
- Add loading spinner/progress bar
- Improve README documentation
- Add sample images for testing
- Fix bugs and broken links

---

## 📂 Branch Naming Guidelines

Use clear branch names:

```bash
feature/feature-name
fix/bug-name
docs/update-readme
ui/improve-layout
```

Examples:

```bash
feature/csv-export
fix/image-upload-error
docs/add-contributing-guide
ui/improve-result-section
```

---

## ✍️ Commit Message Guidelines

Write short and meaningful commit messages.

Good examples:

```bash
git commit -m "Add CSV export for batch predictions"
git commit -m "Fix image upload validation"
git commit -m "Improve Streamlit result UI"
git commit -m "Add Grad-CAM visualization"
```

Avoid vague messages like:

```bash
git commit -m "changes"
git commit -m "update"
git commit -m "fixed stuff"
```

---

## 🧪 Testing Guidelines

Before submitting a PR:

- Run the test suite and make sure it passes:

```bash
python -m pytest tests -q
```

  `tests/test_real_model.py` uses the real bundled model. Its accuracy and API
  checks also need the labelled test faces; create them with
  `python scripts/prepare_dataset.py --train-shards 0 --val-shards 0`.
- Never replace real model output with mocked or random predictions outside of `tests/`.
- If you retrain the model, re-run `python evaluate.py --test-dir data/faces/test`
  and commit the regenerated `metrics_cache.json` and `deepfake_detection_model.json`.
- Run the Streamlit app locally
- Upload sample images and check predictions
- Make sure no errors appear in the terminal
- Test your feature with different image formats if applicable
- Check that the UI works properly

Run the app using:

```bash
streamlit run app.py
```

---

## 📌 Issue Assignment Rules

Please do not start working on an issue until it is assigned to you.

To request assignment, comment on the issue:

```md
I would like to work on this issue under GSSoC'26. Please assign it to me.
```

Once assigned, try to complete the work within the given timeline.

If you are unable to complete it, please inform the project admin or maintainers.

---

## ⏳ Timeline

For assigned issues, contributors should follow the timeline mentioned by the project admin or maintainer.

If no timeline is mentioned, try to raise a Pull Request within a reasonable time.

---

## 📷 Screenshots and Demo

For UI-related changes, please add screenshots or a short screen recording in your PR.

This helps maintainers review your contribution faster.

---

## ❌ What Not to Do

Please avoid:

- Working on unassigned issues
- Creating spam PRs
- Making unrelated changes
- Changing the entire project structure without discussion
- Copy-pasting code without understanding it
- Submitting code that does not run
- Creating multiple PRs for the same issue

---

## 📜 Code of Conduct

Please be respectful and supportive toward all contributors.

We do not allow:

- Rude or disrespectful comments
- Spam
- Harassment
- Discrimination
- Unprofessional behavior

Let’s keep the community friendly and helpful.

---

## 🙋 Need Help?

If you have any questions, feel free to ask in the issue discussion.

Project admins and maintainers will help you.
