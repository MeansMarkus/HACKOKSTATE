// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyB_oF9aOSx4lv82OR_1E80ICpVkuFSaqec",
  authDomain: "parking-helper-8e114.firebaseapp.com",
  projectId: "parking-helper-8e114",
  storageBucket: "parking-helper-8e114.firebasestorage.app",
  messagingSenderId: "39886025147",
  appId: "1:39886025147:web:9050f2c4c69fe0e6a0549e",
  measurementId: "G-4GX2PHD1B3"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);