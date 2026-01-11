# MedivisionAI 🏥

> **AI-Powered Healthcare Early Screening Web Application**

MedivisionAI is a simple, ethical, and user-first web application that leverages artificial intelligence to provide early screening insights from medical X-ray images. Our mission is to make healthcare more accessible, especially for users in remote regions, while maintaining the highest standards of medical ethics.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![React](https://img.shields.io/badge/React-18.x-61dafb.svg)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16.x-339933.svg)](https://nodejs.org/)

---

## 🌟 Features

### Core Functionality
- **🔍 AI-Powered X-Ray Analysis**: Upload X-ray images for instant AI-based screening
- **📊 Explainable Results**: Get clear, understandable insights with visual explanations
- **🌐 Multi-Language Support**: Available in English, Hindi, and Gujarati
- **⚡ Fast & Responsive**: Optimized for quick results and smooth user experience
- **🔒 Privacy-First**: Your medical images are processed securely

### User Experience
- **Minimalist Design**: Clean, medical-grade interface that's easy to understand
- **Mobile-Friendly**: Works seamlessly across all devices
- **Accessible**: Designed for users with varying technical expertise
- **Educational**: Provides general health guidance alongside screening results

### Ethical AI
- **⚠️ Clear Disclaimers**: Prominent notices that this is NOT a medical diagnosis
- **🤝 Transparency**: Explainable AI results with confidence scores
- **📋 Responsible Use**: Encourages users to consult healthcare professionals

---

## 🎯 Target Users

- **General Public**: Anyone seeking early health awareness
- **Patients**: Individuals looking for preliminary screening insights
- **Remote Communities**: Users in areas with limited access to healthcare facilities
- **Health-Conscious Individuals**: People interested in proactive health monitoring

---

## 🏗️ Project Structure

```
dau_final/
├── frontend/                 # React-based frontend application
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Application pages
│   │   ├── locales/         # i18n translation files
│   │   ├── services/        # API integration
│   │   └── styles/          # CSS and styling
│   └── public/              # Static assets
│
├── backend/                 # Node.js backend server
│   ├── models/              # AI models and ML logic
│   ├── routes/              # API endpoints
│   ├── controllers/         # Business logic
│   ├── middleware/          # Authentication, validation
│   └── utils/               # Helper functions
│
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v16.x or higher)
- **npm** or **yarn**
- **Python** (v3.8 or higher) - for AI model backend
- **Git**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BhumiPatel1221/dau_final.git
   cd dau_final
   ```

2. **Set up the Backend**
   ```bash
   cd backend
   npm install
   # or
   yarn install
   ```

3. **Set up the Frontend**
   ```bash
   cd ../frontend
   npm install
   # or
   yarn install
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the `backend` directory:
   ```env
   PORT=5000
   NODE_ENV=development
   MONGODB_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   AI_MODEL_PATH=./models/xray_model
   ```

   Create a `.env` file in the `frontend` directory:
   ```env
   REACT_APP_API_URL=http://localhost:5000/api
   REACT_APP_ENV=development
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd backend
   npm run dev
   # Backend will run on http://localhost:5000
   ```

2. **Start the Frontend Development Server**
   ```bash
   cd frontend
   npm start
   # Frontend will run on http://localhost:3000
   ```

3. **Access the Application**
   
   Open your browser and navigate to `http://localhost:3000`

---

## 🔧 Technology Stack

### Frontend
- **React.js** - UI framework
- **react-i18next** - Internationalization
- **Axios** - HTTP client
- **CSS3** - Styling with modern design principles
- **React Router** - Navigation

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **MongoDB** - Database
- **TensorFlow/PyTorch** - AI model inference
- **Multer** - File upload handling
- **JWT** - Authentication

### AI/ML
- **Deep Learning Models** - For X-ray image analysis
- **Computer Vision** - Image processing and feature extraction
- **Explainable AI** - Grad-CAM for visual explanations

---

## 📖 Usage Guide

### Uploading an X-Ray Image

1. **Navigate to the Upload Section**
   - Click on "Upload X-Ray" from the home page

2. **Select Your Image**
   - Choose an X-ray image from your device
   - Supported formats: JPG, PNG, DICOM

3. **Select Body Part**
   - Choose the relevant body part (Chest, Hand, Leg, etc.)

4. **Submit for Analysis**
   - Click "Analyze" to process the image

5. **View Results**
   - Review the AI screening results
   - Check the confidence score and visual explanations
   - Read the health guidance provided

### Understanding Results

- **NORMAL**: No significant abnormalities detected
- **FRACTURE DETECTED**: Potential fracture identified
- **ABNORMALITY DETECTED**: Other abnormalities found
- **Confidence Score**: Indicates the AI's certainty (0-100%)
- **Visual Explanation**: Highlighted areas of interest on the X-ray

---

## ⚠️ Important Disclaimer

**THIS APPLICATION DOES NOT PROVIDE MEDICAL DIAGNOSIS OR TREATMENT**

MedivisionAI is an early screening tool designed to provide preliminary insights only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. 

**Always consult with qualified healthcare professionals for:**
- Medical diagnosis
- Treatment decisions
- Health concerns
- Follow-up care

The AI screening results should be used as a supplementary tool to encourage timely medical consultation, not as a definitive medical assessment.

---

## 🌍 Multi-Language Support

MedivisionAI supports the following languages:

- **English** (en)
- **हिंदी** (hi) - Hindi
- **ગુજરાતી** (gu) - Gujarati

All user-facing content, including UI elements, instructions, and results, is fully translated.

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines
- Follow the existing code style
- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

---

## 🐛 Bug Reports & Feature Requests

Found a bug or have a feature request? Please open an issue on GitHub:

1. Go to the [Issues](https://github.com/BhumiPatel1221/dau_final/issues) page
2. Click "New Issue"
3. Provide a clear description with steps to reproduce (for bugs)
4. Add relevant labels

---


---

## 👥 Team

**MedivisionAI** is developed and maintained by passionate developers committed to making healthcare more accessible through technology.

- **Project Lead**: Bhumi Patel
- **GitHub**: [@BhumiPatel1221](https://github.com/BhumiPatel1221)

---



---

## 📞 Contact & Support

For questions, support, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/BhumiPatel1221/dau_final/issues)
-
- **Project Repository**: [https://github.com/BhumiPatel1221/dau_final](https://github.com/BhumiPatel1221/dau_final)

---

## 🔮 Roadmap


---

## 📊 Project Status

**Status**: Active Development 🚧

This project is actively being developed and improved. We're working towards a stable v1.0 release.

---

<div align="center">


</div>
