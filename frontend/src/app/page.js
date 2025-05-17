"use client";
import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import Particles from "react-tsparticles";
import { useCallback } from "react";
import { FaPaperPlane, FaUpload, FaRobot, FaUser } from "react-icons/fa";
import { loadSlim } from "tsparticles-slim";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    setMessages([
      { sender: "bot", text: "🤖 Please upload your Excel file to get started." },
    ]);
  }, []);

  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine); // Use loadSlim instead of loadFull
  }, []);


  const particlesLoaded = useCallback(async (container) => {
    // Optional: Add any logic to run after particles are loaded
  }, []);

  // --- PARTICLES OPTIONS (interactive, white, subtle) ---
  const particlesOptions = {
    background: { color: "#111" },
    fpsLimit: 60,
    interactivity: {
      events: {
        onHover: { enable: true, mode: "repulse" },
        onClick: { enable: true, mode: "push" },
        resize: true,
      },
      modes: {
        repulse: { distance: 90, duration: 0.4 },
        push: { quantity: 2 },
      },
    },
    particles: {
      color: { value: "#fff" },
      links: {
        enable: true,
        color: "#fff",
        distance: 120,
        opacity: 0.18,
        width: 1,
      },
      move: {
        enable: true,
        speed: 0.6,
        direction: "none",
        outModes: { default: "bounce" },
      },
      number: { value: 45, density: { enable: true, area: 900 } },
      opacity: { value: 0.12 },
      shape: { type: "circle" },
      size: { value: { min: 2, max: 4 } },
    },
    detectRetina: true,
  };

  // --- FILE UPLOAD ---
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check file extension
      const fileExt = file.name.split('.').pop().toLowerCase();
      if (!['xlsx', 'xls', 'csv'].includes(fileExt)) {
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: "❌ Please upload only Excel (.xlsx, .xls) or CSV (.csv) files." },
        ]);
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      setIsUploading(true);
      
      // Show appropriate file type message
      const fileType = fileExt === 'csv' ? 'CSV' : 'Excel';
      setMessages((prev) => [
        ...prev,
        { sender: "user", text: `📄 Uploaded ${fileType} file: ${file.name}` },
      ]);
      try {
        const response = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (response.ok) {
          setMessages((prev) => [
            ...prev,
            { sender: "bot", text: "✅ File uploaded successfully!" },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            { sender: "bot", text: `❌ Error: ${data.error}` },
          ]);
        }
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: `❌ Error: ${error.message}` },
        ]);
      } finally {
        setIsUploading(false);
      }
    }
  };

  // --- SEND MESSAGE ---
  const handleSend = async () => {
    if (input.trim() === "") return;
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setMessages((prev) => [
      ...prev,
      { sender: "bot", text: "Thinking...", isLoading: true },
    ]);
    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();
      setMessages((prev) =>
        prev.map((msg, idx) =>
          idx === prev.length - 1
            ? { ...msg, text: data.response, isLoading: false }
            : msg
        )
      );
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg, idx) =>
          idx === prev.length - 1
            ? { ...msg, text: `❌ Error: ${error.message}`, isLoading: false }
            : msg
        )
      );
    }
  };

  // --- SEND ON ENTER ---
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // --- RENDER ---
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-black">
      <Particles
        id="tsparticles"
        init={particlesInit}
        loaded={particlesLoaded}
        options={particlesOptions}
        className="absolute inset-0 z-0"
      />

      <div className="relative z-10 w-full max-w-6xl p-4">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, type: "spring" }}
          className="bg-[#181818] rounded-2xl shadow-xl border border-[#232323] p-6"
        >
          {/* Header */}
          <div className="flex items-center justify-center mb-6">
            <FaRobot className="text-3xl text-white mr-2" />
            <h1 className="text-3xl font-semibold text-white tracking-tight">
              Excel Assistant
            </h1>
          </div>

          {/* Chat Window */}
          <div className="chat-window h-96 overflow-y-auto flex flex-col gap-3 px-1 py-2 mb-4 scrollbar-thin scrollbar-thumb-[#232323] scrollbar-track-transparent">
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
                className={`flex items-end ${
                  msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {/* Avatar */}
                {msg.sender === "bot" && (
                  <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-[#232323] mr-2">
                    <FaRobot className="text-white" />
                  </div>
                )}
                {/* Bubble */}
                <div
                  className={`max-w-4xl p-4 rounded-xl shadow text-base transition-colors duration-200 ${
                    msg.sender === "user"
                      ? "bg-[#232323] text-white rounded-br-none hover:bg-[#313131]"
                      : "bg-[#181818] border border-[#232323] text-white rounded-bl-none hover:bg-[#232323]"
                  }`}
                >
                  {msg.isLoading ? (
                    <span className="animate-pulse">Thinking...</span>
                  ) : (
                    <div className="whitespace-pre-wrap leading-relaxed">
                      {msg.text.split('\n').map((line, i) => (
                        <React.Fragment key={i}>
                          {line}
                          {i !== msg.text.split('\n').length - 1 && <br />}
                        </React.Fragment>
                      ))}
                    </div>
                  )}
                </div>
                {/* Avatar */}
                {msg.sender === "user" && (
                  <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-[#232323] ml-2">
                    <FaUser className="text-white" />
                  </div>
                )}
              </motion.div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Input Area */}
          <div className="flex items-center bg-[#232323] rounded-xl p-2 shadow-inner">
            {/* File Upload Button */}
            <label
              htmlFor="file-upload"
              className="flex items-center px-3 py-2 bg-[#181818] text-white rounded-xl cursor-pointer mr-3 hover:bg-[#232323] transition duration-200"
              title="Upload Excel File"
            >
              <FaUpload className="mr-2" />
              <span className="text-sm font-medium">
                {isUploading ? "Uploading..." : "Upload Excel/CSV"}
              </span>
              <small className="text-gray-400 text-xs ml-1">(xlsx, xls, csv)</small>
              <input
                type="file"
                id="file-upload"
                onChange={handleUpload}
                className="hidden"
                accept=".xlsx,.xls,.csv"
                disabled={isUploading}
              />
            </label>
            {/* Message Input */}
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              className="flex-grow px-4 py-2 bg-black text-white rounded-xl outline-none placeholder:text-gray-400 focus:ring-2 focus:ring-[#232323] transition"
              disabled={isUploading}
            />
            {/* Send Button */}
            <button
              onClick={handleSend}
              disabled={input.trim() === "" || isUploading}
              className="ml-3 px-4 py-2 flex items-center justify-center bg-[#181818] rounded-xl text-white font-semibold hover:bg-[#232323] transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow"
              title="Send"
            >
              <FaPaperPlane className="mr-1" />
              Send
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
