import React, { useEffect, useState, useRef } from "react";
import { AlertCircle, ChevronDown, Camera, Check, Info, BarChart2, Clock, RefreshCw, Coffee } from "lucide-react";
import { motion, AnimatePresence, useAnimation } from "framer-motion";
import { useInView } from "react-intersection-observer";


const Analysis = () => {
  const [feedback, setFeedback] = useState([]);
  const [overview, setOverview] = useState({
    postureScore: 0,
    timeTracked: "0.0 hrs",
    corrections: 0,
    breaksTaken: 0,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [showFeedback, setShowFeedback] = useState(true);
  const [cameraError, setCameraError] = useState(false);
  const [animateScore, setAnimateScore] = useState(false);
  const videoRef = useRef(null);
  const controls = useAnimation();
  const [ref, inView] = useInView({
    threshold: 0.2,
    triggerOnce: false
  });

  // Updated color palette to match Home page colors
  const colors = {
    bgGradient: "from-[#171413] to-[#1f1a18]", // dark earthy gradient background
    cardBg: "bg-[#251c17]",                     // Home cards background dark brown
    headingGradient: "text-[#C96442]",           // ACCENT warm brown-red
    buttonGradient: "bg-gradient-to-r from-[#C96442] to-[#b4563f]", // ACCENT button gradient
    accent: "#C96442",                          // ACCENT color
    text: "text-[#eee]",                        // TEXT_LIGHT main text color
    textSecondary: "text-[#aaa]",               // TEXT_FADED secondary text
    success: "text-green-400",                   // Muted green for success
    warning: "text-[#e94e20]",                   // Warm warning orange-red
    error: "text-red-500",                       // Keep red for errors
  };


  // Function to fetch feedback from the Flask backend
  const fetchFeedback = async () => {
    try {
      // Avoid flickering loader by not setting isLoading every poll.
      const response = await fetch("http://localhost:5000/feedback");
      const data = await response.json();
      setFeedback(data.feedback);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching feedback:", error);
      setIsLoading(false);
    }
  };


  // Function to fetch today's overview data
  const fetchOverview = async () => {
    try {
      const res = await fetch("http://localhost:5000/overview");
      const data = await res.json();
      
      // Trigger score animation when score changes
      if (data.postureScore !== overview.postureScore) {
        setAnimateScore(true);
        setTimeout(() => setAnimateScore(false), 2000);
      }
      
      setOverview(data);
    } catch (error) {
      console.error("Error fetching overview:", error);
    }
  };


  // Poll feedback every second
  useEffect(() => {
    fetchFeedback();
    const intervalId = setInterval(fetchFeedback, 1000);
    return () => clearInterval(intervalId);
  }, []);


  // Poll overview data every second
  useEffect(() => {
    fetchOverview();
    const overviewInterval = setInterval(fetchOverview, 1000);
    return () => clearInterval(overviewInterval);
  }, []);


  // Start animations when components come into view
  useEffect(() => {
    if (inView) {
      controls.start("visible");
    }
  }, [controls, inView]);


  // Function to determine feedback item color based on content
  const getFeedbackColor = (item) => {
    if (item.toLowerCase().includes("correct") || item.toLowerCase().includes("good")) {
      return colors.success;   // using muted green from Home
    } else if (item.toLowerCase().includes("warning") || item.toLowerCase().includes("caution")) {
      return colors.warning;   // warm orange-red from Home
    } else if (item.toLowerCase().includes("error") || item.toLowerCase().includes("incorrect")) {
      return colors.error;     // red for errors
    }
    return colors.accent;      // ACCENT color for others
  };


  const getFeedbackBgColor = (item) => {
    if (item.toLowerCase().includes("correct") || item.toLowerCase().includes("good")) {
      return "bg-green-50"; // Keeping light green bg for success - no Home equivalent provided
    } else if (item.toLowerCase().includes("warning") || item.toLowerCase().includes("caution")) {
      return "bg-[rgba(233,78,32,0.1)]"; // translucent warm warning bg
    } else if (item.toLowerCase().includes("error") || item.toLowerCase().includes("incorrect")) {
      return "bg-red-50"; // keep light red bg for errors
    }
    return "bg-[rgba(201,100,66,0.1)]";  // faint warm accent background
  };


  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };


  const floatingAnimation = {
    initial: { y: 0 },
    animate: { 
      y: [-5, 5, -5],
      transition: {
        duration: 3,
        repeat: Infinity,
        repeatType: "loop",
        ease: "easeInOut"
      }
    }
  };


  const pulseAnimation = {
    initial: { scale: 1 },
    animate: { 
      scale: [1, 1.05, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        repeatType: "loop"
      }
    }
  };


  return (
    <div className="w-full overflow-x-hidden scroll-smooth">
      <div className={`min-h-screen bg-gradient-to-br ${colors.bgGradient} pt-24 pb-12`}>
        <div className="container mx-auto px-6">
          {/* Page Title with Animation */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h1 className={`text-3xl md:text-4xl font-bold ${colors.text} mb-2`}>
              Posture Analysis
            </h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 1 }}
              className={`text-xl bg-clip-text text-transparent bg-gradient-to-r from-[${colors.accent}] to-[#b4563f]`}
            >
              Real-time feedback to improve your sitting habits
            </motion.p>
          </motion.div>


          <div className="flex flex-col md:flex-row gap-8">
            {/* Camera Feed Section */}
            <motion.div 
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="flex-1"
            >
              <div className={`${colors.cardBg} rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300`}>
                <div className="bg-gradient-to-r from-[#C96442] to-[#b4563f] p-4 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Camera className="h-5 w-5 text-white" />
                    <h2 className="text-white font-medium">Camera Feed</h2>
                   </div>
                   <motion.div 
                     {...pulseAnimation}
                     className="flex items-center"
                   >
                     <div className="h-2 w-2 rounded-full bg-green-400 mr-2"></div>
                     <span className="text-gray-100 text-sm">Live</span>
                   </motion.div>
                </div>
                <div className="relative aspect-video bg-[rgba(201,100,66,0.1)] flex items-center justify-center overflow-hidden rounded-b-xl">
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    className="w-full h-full relative"
                  >
                    <img 
                      ref={videoRef}
                      src="http://localhost:5000/video_feed_front"
                      alt="Camera feed" 
                      className="w-full h-full object-contain"
                      onError={() => setCameraError(true)}
                    />
                    {/* Position visualization overlay */}
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.5 }}
                      transition={{ delay: 1.2, duration: 0.8 }}
                      className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    >
                      {!cameraError && !isLoading && (
                        <svg className="w-full h-full" viewBox="0 0 1000 800" xmlns="http://www.w3.org/2000/svg">
                          {/* Center vertical line for alignment */}
                          <motion.line 
                            x1="500" y1="0" x2="500" y2="800" 
                            stroke={colors.accent} strokeWidth="1" strokeDasharray="5,5"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{ duration: 1.5, delay: 0.5 }}
                          />
                          
                          {/* Shoulder level guide */}
                          <motion.line 
                            x1="300" y1="200" x2="700" y2="200" 
                            stroke={colors.accent} strokeWidth="1" strokeDasharray="5,5"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{ duration: 1, delay: 0.8 }}
                          />
                          
                          {/* Animated target area for head */}
                          <motion.circle 
                            cx="500" cy="100" r="80" 
                            fill="none" stroke={colors.accent} strokeWidth="2" strokeDasharray="5,5"
                            initial={{ scale: 0.5, opacity: 0 }}
                            animate={{ scale: 1, opacity: 0.5 }}
                            transition={{ duration: 1, delay: 1 }}
                          />
                        </svg>
                      )}
                    </motion.div>
                  </motion.div>
                  
                  {isLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-[#1f1a18] bg-opacity-60 backdrop-blur-sm rounded-b-xl">
                      <motion.div
                        animate={{ 
                          rotate: 360,
                          borderRadius: ["20%", "20%", "50%", "50%", "20%"]
                        }}
                        transition={{ 
                          rotate: { duration: 1.5, repeat: Infinity, ease: "linear" },
                          borderRadius: { duration: 3, repeat: Infinity }
                        }}
                        className={`w-16 h-16 border-4 border-[${colors.accent}] border-t-transparent rounded-full`}
                      />
                    </div>
                  )}
                  
                  {cameraError && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#1f1a18] bg-opacity-30 backdrop-blur-sm rounded-b-xl">
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5 }}
                      >
                        <Camera className={`h-16 w-16 text-[${colors.accent}] mb-4`} />
                      </motion.div>
                      <motion.p 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, duration: 0.5 }}
                        className={`text-[#C96442] text-xl font-medium mb-2`}
                      >
                        Camera feed unavailable
                      </motion.p>
                      <motion.p 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                        className={`${colors.textSecondary} text-sm text-center max-w-xs`}
                      >
                        Please ensure the Flask server is running at http://localhost:5000
                      </motion.p>
                      <motion.button
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        transition={{ delay: 0.7, duration: 0.5 }}
                        className={`mt-6 ${colors.accent} text-white px-6 py-2 rounded-lg font-medium flex items-center`}
                      >
                        <RefreshCw size={16} className="mr-2" /> Try Again
                      </motion.button>
                    </div>
                  )}
                </div>
                <div className={`${colors.cardBg} p-4`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <motion.div 
                        animate={{
                          scale: [1, 1.2, 1],
                          backgroundColor: ["#10b981", colors.accent, "#10b981"]
                        }}
                        transition={{ 
                          duration: 2,
                          repeat: Infinity,
                          repeatType: "reverse" 
                        }}
                        className="w-3 h-3 rounded-full bg-green-500"
                      ></motion.div>
                      <span className={`${colors.textSecondary} text-sm`}>Analysis active</span>
                    </div>
                    <motion.span 
                      whileHover={{ scale: 1.05 }}
                      className={`text-xs ${colors.accent} font-medium px-2 py-1 bg-[rgba(201,100,66,0.1)] rounded-full`}
                    >
                      Auto-refresh enabled
                    </motion.span>
                  </div>
                </div>
              </div>
            </motion.div>


            {/* Right Side Panel */}
            <motion.div 
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="w-full md:w-96 flex flex-col gap-6"
              ref={ref}
            >
              {/* Feedback Section */}
              <motion.div 
                variants={containerVariants}
                initial="hidden"
                animate={controls}
                className={`${colors.cardBg} rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300`}
              >
                <div 
                  className="bg-gradient-to-r from-[#C96442] to-[#b4563f] p-4 flex items-center justify-between cursor-pointer"
                  onClick={() => setShowFeedback(!showFeedback)}
                >
                  <div className="flex items-center">
                    <AlertCircle className="h-5 w-5 mr-3 text-white" />
                    <h2 className="text-white font-medium">Posture Feedback</h2>
                  </div>
                  <motion.div
                    animate={{ rotate: showFeedback ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <ChevronDown className="h-5 w-5 text-white" />
                  </motion.div>
                </div>
                
                <AnimatePresence>
                  {showFeedback && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.4, ease: "easeInOut" }}
                      className="overflow-hidden"
                    >
                      {feedback.length === 0 ? (
                        <motion.div 
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.2, duration: 0.5 }}
                          className={`${colors.cardBg} flex items-center justify-center py-16 text-center`}
                        >
                          <div className="flex flex-col items-center">
                            <motion.div
                              {...floatingAnimation}
                              className="w-20 h-20 bg-[rgba(201,100,66,0.1)] rounded-full flex items-center justify-center mb-4"
                            >
                              <Camera className={`h-10 w-10 text-[${colors.accent}]`} />
                            </motion.div>
                            <p className={`${colors.text} font-medium mb-2`}>
                              No feedback available at the moment
                            </p>
                            <p className={`${colors.textSecondary} text-sm max-w-xs`}>
                              Please ensure you're in frame and your posture is visible
                            </p>
                          </div>
                        </motion.div>
                      ) : (
                        <div className="p-4 max-h-96 overflow-y-auto">
                          <ul className="space-y-3">
                            {feedback.map((item, index) => (
                              <motion.li
                                key={index}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ duration: 0.5, delay: index * 0.1 }}
                                className={`flex items-start p-3 rounded-lg ${getFeedbackBgColor(item)} border border-opacity-20 ${item.toLowerCase().includes("correct") ? "border-green-200" : item.toLowerCase().includes("warning") ? "border-[rgba(233,78,32,0.5)]" : "border-red-200"}`}
                              >
                                <div className="mr-3 mt-0.5">
                                  {item.toLowerCase().includes("correct") || item.toLowerCase().includes("good") ? (
                                    <Check size={18} className={colors.success} />
                                  ) : (
                                    <Info size={18} className={getFeedbackColor(item)} />
                                  )}
                                </div>
                                <span className={`${colors.text} `}>{item}</span>
                              </motion.li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className={`${colors.cardBg} p-3 border-t border-[rgba(201,100,66,0.2)] flex justify-between items-center`}>
                        <span className={`${colors.textSecondary} text-xs`}>
                          Last updated: {new Date().toLocaleTimeString()}
                        </span>
                        <motion.span 
                          animate={{ 
                            scale: feedback.length > 0 ? [1, 1.1, 1] : 1
                          }}
                          transition={{ 
                            duration: 0.5, 
                            repeat: feedback.length > 0 ? 1 : 0
                          }}
                          className={`text-xs font-medium px-2 py-1 rounded-full bg-[rgba(201,100,66,0.1)] text-[${colors.accent}]`}
                        >
                          {feedback.length} items
                        </motion.span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
              
              {/* Today's Overview Section */}
              <motion.div 
                variants={containerVariants}
                initial="hidden"
                animate={controls}
                className={`${colors.cardBg} rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300 overflow-hidden`}
              >
                <div className="bg-gradient-to-r from-[#C96442] to-[#b4563f] p-4 flex items-center">
                  <BarChart2 className="h-5 w-5 mr-3 text-white" />
                  <h3 className="text-white font-medium">Today's Overview</h3>
                </div>
                
                <div className="p-4 grid grid-cols-2 gap-4">
                  <motion.div 
                    variants={itemVariants}
                    className="bg-[rgba(201,100,66,0.1)] p-4 rounded-lg relative overflow-hidden group"
                  >
                    <motion.div
                      animate={animateScore ? {
                        opacity: [0, 1, 0],
                        scale: [1, 1.5, 1],
                        y: [0, -20, 0]
                      } : {}}
                      transition={{ duration: 1 }}
                      className={`absolute right-2 top-2 text-[${colors.accent}] font-bold text-xs`}
                    >
                      {overview.postureScore > 50 ? "++" : "--"}
                    </motion.div>
                    <p className={`${colors.textSecondary} text-xs mb-1 flex items-center`}>
                      <BarChart2 className={`h-3 w-3 mr-1 text-[${colors.accent}]`} /> Posture Score
                    </p>
                    <div className="flex items-end space-x-1">
                      <motion.p 
                        animate={animateScore ? {
                          scale: [1, 1.2, 1],
                          color: [
                            colors.accent,
                            "#b4563f",
                            colors.accent
                          ]
                        } : {}}
                        transition={{ duration: 1 }}
                        className={`text-[${colors.accent}] text-2xl font-bold`}
                      >
                        {overview.postureScore}%
                      </motion.p>
                    </div>
                    {/* Progress bar */}
                    <div className="w-full h-2 bg-[rgba(201,100,66,0.2)] rounded-full mt-2 overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${overview.postureScore}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className={`h-full bg-gradient-to-r from-[#C96442] to-[#b4563f] rounded-full`}
                      />
                    </div>
                  </motion.div>
                  
                  <motion.div 
                    variants={itemVariants}
                    className="bg-[rgba(201,100,66,0.05)] p-4 rounded-lg"
                  >
                    <p className={`${colors.textSecondary} text-xs mb-1 flex items-center`}>
                      <Clock className={`h-3 w-3 mr-1 text-[${colors.accent}]`} /> Time Tracked
                    </p>
                    <p className={`text-[${colors.accent}] text-2xl font-bold`}>{overview.timeTracked}</p>
                    <motion.div 
                      animate={{
                        opacity: [0.5, 1, 0.5],
                      }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="w-full h-1 bg-[rgba(201,100,66,0.2)] rounded-full mt-3"
                    />
                  </motion.div>
                  
                  <motion.div 
                    variants={itemVariants}
                    className="bg-pink-50 p-4 rounded-lg"
                  >
                    <p className={`${colors.textSecondary} text-xs mb-1 flex items-center`}>
                      <RefreshCw className="h-3 w-3 mr-1" /> Corrections
                    </p>
                    <div className="flex items-baseline">
                      <motion.p 
                        animate={overview.corrections > 0 ? {
                          scale: [1, 1.1, 1]
                        } : {}}
                        transition={{ duration: 0.3 }}
                        className="text-pink-600 text-2xl font-bold"
                      >
                        {overview.corrections}
                      </motion.p>
                      <span className="text-pink-400 text-sm ml-1">times</span>
                    </div>
                    <div className="mt-2 flex space-x-1">
                      {[...Array(5)].map((_, i) => (
                        <motion.div 
                          key={i}
                          animate={{
                            opacity: i < Math.min(5, overview.corrections) ? 1 : 0.3
                          }}
                          className="w-full h-1 bg-pink-300 rounded-full"
                        />
                      ))}
                    </div>
                  </motion.div>
                  
                  <motion.div 
                    variants={itemVariants}
                    className="bg-blue-50 p-4 rounded-lg"
                  >
                    <p className={`${colors.textSecondary} text-xs mb-1 flex items-center`}>
                      <Coffee className="h-3 w-3 mr-1" /> Breaks Taken
                    </p>
                    <div className="flex items-baseline">
                      <motion.p 
                        animate={overview.breaksTaken > 0 ? {
                          scale: [1, 1.1, 1]
                        } : {}}
                        transition={{ duration: 0.3 }}
                        className="text-blue-600 text-2xl font-bold"
                      >
                        {overview.breaksTaken}
                      </motion.p>
                    </div>
                    <div className="flex space-x-1 mt-2">
                      {[...Array(3)].map((_, i) => (
                        <motion.div 
                          key={i}
                          initial={{ scale: 0.8, opacity: 0.3 }}
                          animate={{ 
                            scale: i < overview.breaksTaken ? 1 : 0.8,
                            opacity: i < overview.breaksTaken ? 1 : 0.3
                          }}
                          className="w-6 h-6 rounded-full flex items-center justify-center bg-blue-100"
                        >
                          <Coffee size={12} className="text-blue-500" />
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                </div>
              </motion.div>
              
              {/* Quick Tips Section */}
              <motion.div 
                variants={containerVariants}
                initial="hidden"
                animate={controls}
                whileHover={{ y: -5 }}
                transition={{ duration: 0.3 }}
                className={`${colors.cardBg} rounded-xl shadow-lg overflow-hidden`}
              >
                <div className="bg-gradient-to-r from-[#b4563f] to-[#e94e20] p-4 flex items-center">
                  <Info className="h-5 w-5 mr-3 text-white" />
                  <h3 className="text-white font-medium">Posture Quick Tips</h3>
                </div>
                <div className="p-5">
                  <ul className={`${colors.text} space-y-3`}>
                    {[
                      { tip: "Keep your head aligned with your spine", icon: "🧠" },
                      { tip: "Relax your shoulders and avoid hunching", icon: "💪" },
                      { tip: "Keep your screen at eye level", icon: "👁️" },
                      { tip: "Take breaks every 30 minutes", icon: "⏰" },
                      { tip: "Maintain 90° angle at your elbows", icon: "💻" }
                    ].map((item, i) => (
                      <motion.li 
                        key={i}
                        variants={itemVariants}
                        className={`flex items-center bg-[rgba(201,100,66,0.1)] p-2 px-3 rounded-lg`}
                        whileHover={{ x: 5 }}
                      >
                        <motion.span 
                          className="mr-3 text-lg"
                          animate={{ rotate: [-5, 5, -5] }}
                          transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
                        >
                          {item.icon}
                        </motion.span>
                        {item.tip}
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            </motion.div>
          </div>
          
          {/* Bottom Call to Action */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            viewport={{ once: true }}
            className="mt-12 text-center"
          >
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: `0 10px 25px -5px rgba(201,100,66,0.6)` }}
              whileTap={{ scale: 0.95 }}
              className="bg-gradient-to-r from-[#C96442] to-[#b4563f] text-white px-8 py-4 rounded-lg font-bold text-lg shadow-lg inline-flex items-center"
            >
              View Detailed Report
              <motion.span
                animate={{ x: [0, 5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, repeatType: "loop" }}
                className="ml-2"
              >
                →
              </motion.span>
            </motion.button>
            <motion.p 
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ delay: 0.8, duration: 0.5 }}
              viewport={{ once: true }}
              className={`${colors.textSecondary} mt-4`}
            >
              Get a comprehensive analysis of your posture habits and personalized recommendations
            </motion.p>
          </motion.div>
        </div>
      </div>
    </div>
  );
};


export default Analysis;
