import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// --- Custom Sunburst Logo from your attachment ---
const SunburstLogo = ({size = 32}) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
    <g transform="translate(16,16)">
      {/* Center circle */}
      <circle cx="0" cy="0" r="3" fill="#ff8c42"/>
      {/* Radiating lines */}
      <path d="M0,-12 L0,-8" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M8.5,-8.5 L6,-6" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M12,0 L8,0" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M8.5,8.5 L6,6" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M0,12 L0,8" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M-8.5,8.5 L-6,6" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M-12,0 L-8,0" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M-8.5,-8.5 L-6,-6" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
      <path d="M6,-10.4 L4.2,-7.8" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M10.4,-6 L7.8,-4.2" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M10.4,6 L7.8,4.2" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M6,10.4 L4.2,7.8" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M-6,10.4 L-4.2,7.8" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M-10.4,6 L-7.8,4.2" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M-10.4,-6 L-7.8,-4.2" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M-6,-10.4 L-4.2,-7.8" stroke="#ff8c42" strokeWidth="1.5" strokeLinecap="round"/>
    </g>
  </svg>
);

const ChatBot = () => {
  // --- State (both hover and click functionality) ---
  const [isHovered, setIsHovered] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // --- Hover handlers ---
  const handleMouseEnter = () => setIsHovered(true);
  const handleMouseLeave = () => setIsHovered(false);
  
  // --- Click handler ---
  const handleClick = () => setIsClicked(prev => !prev);

  // --- Determine if window should be shown (hover OR clicked) ---
  const shouldShowWindow = isHovered || isClicked;

  // --- Black & Orange Theme Markdown Formatter ---
  const formatMessage = (text) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong style="color:#ff8c42">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em style="color:#ffb366">$1</em>')
      .replace(/\n/g, '<br/>');
  };

  // --- Scroll on new message ---
  useEffect(() => {
    if (shouldShowWindow) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      inputRef.current?.focus();
    }
  }, [messages, shouldShowWindow]);

  // --- Message Send Handler ---
  const getCurrentTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    setMessages((prev) => [
      ...prev,
      { sender: "user", text: inputMessage, timestamp: getCurrentTime() },
    ]);
    setIsLoading(true);
    const userMessage = inputMessage;
    setInputMessage("");
    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: formatMessage(data.response),
          timestamp: getCurrentTime(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "🚨 <strong style='color:#ff8c42'>Connection Error</strong>: Unable to reach the neural network",
          timestamp: getCurrentTime(),
        },
      ]);
    }
    setIsLoading(false);
  };

  // --- Animations ---
  const containerVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.97 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { type: "spring", stiffness: 180, damping: 18 },
    },
    exit: { opacity: 0, y: 8, scale: 0.97, transition: { duration: 0.19 } },
  };
  const messageVariants = {
    hidden: { opacity: 0, x: 40, scale: 0.96 },
    visible: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: { type: "spring", stiffness: 200, damping: 17 },
    },
  };
  const bubbleVariants = {
    hover: { scale: 1.07 },
    tap: { scale: 0.97 },
  };

  // --- Main Render ---
  return (
    <div 
      className="fixed bottom-5 right-5 z-50"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Floating Chat Button - Hover to Preview, Click to Stay */}
      <motion.div
        onClick={handleClick}
        className="w-14 h-14 rounded-full bg-[#1a1a1a] border-2 border-[#ff8c42] shadow-lg flex items-center justify-center cursor-pointer transition-all"
        variants={bubbleVariants}
        whileHover="hover"
        whileTap="tap"
        style={{
          marginBottom: shouldShowWindow ? 26 : 0,
          transition: 'margin-bottom 0.22s cubic-bezier(.77,0,.18,1)'
        }}
      >
        <SunburstLogo size={32} />
      </motion.div>

      {/* Chatbot Window - Shows on Hover OR Click */}
      <AnimatePresence>
        {shouldShowWindow && (
          <motion.div
            key="chatbot"
            className="absolute bottom-20 sm:bottom-[72px] right-0 sm:right-0 flex flex-col w-[320px] sm:w-[350px] h-[360px] sm:h-[400px]
              rounded-3xl shadow-2xl border-2 border-[#ff8c42] bg-[#1a1a1a] overflow-hidden"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={containerVariants}
            style={{
              zIndex: 51
            }}
          >
            {/* Header - Black & Orange Theme with Smaller Font */}
            <div className="flex items-center gap-3 px-4 py-3 border-b-2 border-[#ff8c42] bg-[#0d0d0d]">
              <span className="w-6 h-6"><SunburstLogo size={22} /></span>
              <div>
                <span className="font-bold text-[#ffffff] text-sm">Neural</span>
                <span className="ml-2 px-2 py-1 bg-[#ff8c42] text-[#000000] rounded text-[10px] font-bold">AI</span>
                <span className="ml-2 px-2 text-[10px] text-[#ff8c42] animate-pulse">● Online</span>
              </div>
              {isClicked && (
                <div className="ml-auto">
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      setIsClicked(false);
                    }}
                    className="p-1 hover:bg-[#ff8c42]/20 rounded-lg transition-colors"
                  >
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                      <path d="M12 4L4 12M4 4l8 8" stroke="#ff8c42" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
              )}
            </div>

            {/* Messages - Black Background with Orange Accents and Smaller Font */}
            <div className="flex-1 p-4 overflow-y-auto space-y-3" style={{background:"#1a1a1a"}}>
              <div className="text-center pt-1 pb-3">
                <span className="px-3 py-1 text-[10px] bg-[#ff8c42] text-[#000000] rounded-full font-medium">Neural AI v3.0</span>
              </div>
              {messages.map((msg, index) => (
                <motion.div
                  key={index}
                  className={`
                    group relative px-3 py-2 rounded-xl max-w-[85%] break-words
                    ${msg.sender === "user"
                      ? "ml-auto bg-[#ff8c42] text-[#000000] border border-[#ffb366]"
                      : "mr-auto bg-[#2a2a2a] text-[#ffffff] border border-[#ff8c42]"
                    }
                  `}
                  initial="hidden"
                  animate="visible"
                  variants={messageVariants}
                  style={{
                    fontSize:"12px",
                    lineHeight:"1.4"
                  }}
                >
                  {/* User text/plain, bot is html/markdown */}
                  {msg.sender === "user" ? (
                    <span>{msg.text}</span>
                  ) : (
                    <div
                      dangerouslySetInnerHTML={{ __html: msg.text }}
                      style={{ wordBreak: "break-word" }}
                    />
                  )}
                  <div className={`text-[10px] mt-1 flex items-center justify-end ${
                    msg.sender === "user" ? "text-[#00000080]" : "text-[#ffffff80]"
                  }`}>
                    <span>{msg.timestamp}</span>
                  </div>
                </motion.div>
              ))}
              {isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center space-x-2 ml-2"
                >
                  <div className="flex space-x-[2px]">
                    {[...Array(3)].map((_, i) => (
                      <div
                        key={i}
                        className="w-2 h-2 bg-[#ff8c42] rounded-full animate-bounce"
                        style={{ animationDelay: `${i * 0.12}s` }}
                      />
                    ))}
                  </div>
                  <span className="text-[10px] text-[#ff8c42]">Processing…</span>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Bar - Black & Orange Theme with Smaller Font */}
            <form
              onSubmit={sendMessage}
              className="flex items-center space-x-2 p-3 border-t-2 border-[#ff8c42] bg-[#0d0d0d]"
            >
              <input
                type="text"
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask the Neural Network…"
                className="flex-1 px-3 py-2 text-xs rounded-md border-2 border-[#ff8c42] focus:outline-none focus:ring-2 focus:ring-[#ff8c42] bg-[#1a1a1a] text-[#ffffff] placeholder:text-[#999999]"
                disabled={isLoading}
                autoComplete="off"
              />
              <button
                type="submit"
                disabled={isLoading}
                className={`px-3 py-2 bg-[#ff8c42] hover:bg-[#ffb366] text-[#000000] rounded-lg font-medium transition-all disabled:opacity-50`}
              >
                <svg
                  width={16}
                  className="inline-block"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#000000"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{transform:'rotate(-8deg)',marginRight:-2,marginLeft:-2}}
                >
                  <path d="M4 20l16-8-16-8v6l10 2-10 2v6z"></path>
                </svg>
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ChatBot;
