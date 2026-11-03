"use client";
import { useState } from "react";
import axios from "axios";
import Image from "next/image";

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("video", file);

    setLoading(true);
    setResult(null);

    try {
      const res = await axios.post("http://localhost:5000/predict", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4 bg-black text-[#6c11db] relative">
      
      {/* Top-Left Logo and Titles */}
      <div className="absolute top-6 left-6">
        <div className="flex items-center mb-1">
          <Image src="/ApolloIcon.png" alt="Apollo Logo" width={60} height={60} />
          <span className="ml-3 text-7xl font-normal">Apollo</span>
        </div>
        <p className="ml-[75px] text-lg font-bold text-[#6c11db] -mt-2">Deepfake Video Detection</p>
      </div>



      {/* Centered Upload Card */}
      <div className="bg-gray-900 p-6 rounded-xl shadow-lg w-full max-w-md text-center">
        
        {/* Styled file upload button */}
        <label className="block mb-4 cursor-pointer bg-[#6c11db] text-white px-4 py-2 rounded-lg hover:bg-purple-700 active:scale-95 transition-transform">
          {file ? file.name : "Choose File"}
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>

        {/* Analyze Button */}
        <button
          className="bg-[#6c11db] text-white px-4 py-2 rounded-lg hover:bg-purple-700 active:scale-95 disabled:opacity-50"
          disabled={!file || loading}
          onClick={handleSubmit}
        >
          {loading ? "Analyzing..." : "Analyze Video"}
        </button>

        {/* Results */}
        {result && (
          <div className="mt-6 text-left">
            <p><strong>Prediction:</strong> {result.prediction}</p>
            <p><strong>Probability:</strong> {(result.probability * 100).toFixed(2)}%</p>
          </div>
        )}
      </div>
    </main>
  );
}
