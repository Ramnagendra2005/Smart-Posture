import React, { useEffect } from "react";
import { motion, useAnimation } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { ChevronDown, Monitor, Activity, Users, Award, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

const ACCENT = "#C96442";
const BG_DARK = "#171413";
const BG_DARKER = "#120f0e";
const TEXT_LIGHT = "#eee";
const TEXT_FADED = "#aaa";

const Home = () => {
  return (
    <div className="w-full overflow-x-hidden scroll-smooth" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
      <HeroSection />
      <FeaturesSection />
      <HowItWorksSection />
      <BenefitsSection />
      <CTASection />
      <Footer />
    </div>
  );
};

const HeroSection = () => {
  const navigate = useNavigate();

  return (
    <section
      className="relative flex items-center min-h-screen px-6"
      style={{
        background: `linear-gradient(135deg, ${BG_DARK} 0%, #1f1a18 100%)`,
      }}
    >
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="text-left"
        >
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-4xl md:text-6xl font-extrabold"
            style={{ color: ACCENT, letterSpacing: "2px" }}
          >
            Smart <span style={{ color: "#d65730" }}>Posture</span>
          </motion.h1>

          <motion.h2
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 1 }}
            className="text-2xl md:text-3xl font-semibold mt-6 mb-8 max-w-xl"
            style={{
              color: TEXT_LIGHT,
              fontWeight: 700,
            }}
          >
            A Real-Time <span style={{ color: ACCENT }}>ML-Driven System</span> for Posture Detection,{" "}
            <span style={{ color: ACCENT }}>Analysis</span> and Personalized Exercise Recommendations
          </motion.h2>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            className="text-lg max-w-md"
            style={{ color: TEXT_FADED }}
          >
            Improve your posture, <span style={{ color: ACCENT }}>reduce pain</span>,
            and boost productivity with our <span style={{ color: ACCENT }}>AI-powered</span> posture analysis system.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1, duration: 0.6 }}
          >
            <button
              onClick={() => navigate("/analysis")}
              className="px-8 py-4 rounded-lg font-bold text-lg transition-transform duration-300"
              style={{
                backgroundColor: ACCENT,
                marginTop: "2rem",
                color: TEXT_LIGHT,
              }}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = "#b14a2e")}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = ACCENT)}
            >
              Start Your Analysis
            </button>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="flex justify-center"
        >
          <img
            src="https://imgs.search.brave.com/SmkmUJBZJCYgDoYXy5uf-XiiKp18MHIvexlHlkdAiMk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4t/cHJvZC5tZWRpY2Fs/bmV3c3RvZGF5LmNv/bS9jb250ZW50L2lt/YWdlcy9hcnRpY2xl/cy8zMjEvMzIxODYz/L2NvcnJlY3Qtc2l0/dGluZy1wb3N0dXJl/LWRpYWdyYW0tYXQt/YS1jb21wdXRlci1k/ZXNrLmpwZw"
            alt="Correct sitting posture"
            className="rounded-lg max-w-full max-h-96 object-contain"
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.8 }}
          transition={{ delay: 2, duration: 1, repeat: Infinity, repeatType: "reverse" }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
          style={{ color: ACCENT }}
        >
          <ChevronDown size={36} />
        </motion.div>
      </div>
    </section>
  );
};

const FeaturesSection = () => {
  const controls = useAnimation();
  const [ref, inView] = useInView({
    threshold: 0.25,
    triggerOnce: false,
  });

  useEffect(() => {
    if (inView) controls.start("visible");
  }, [controls, inView]);

  const features = [
    {
      icon: <Monitor size={48} color={ACCENT} />,
      title: "Real-Time Detection",
      description: (
        <>
          Instantly analyze your sitting posture through your webcam with{" "}
          <span style={{ color: ACCENT }}>advanced ML algorithms</span>
        </>
      ),
    },
    {
      icon: <Activity size={48} color={ACCENT} />,
      title: "Detailed Analysis",
      description: (
        <>
          Get <span style={{ color: ACCENT }}>comprehensive insights</span> about your posture patterns and potential improvements
        </>
      ),
    },
    {
      icon: <Users size={48} color={ACCENT} />,
      title: "Personalized Recommendations",
      description: (
        <>
          Receive <span style={{ color: ACCENT }}>customized exercise suggestions</span> based on your unique posture profile
        </>
      ),
    },
  ];

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 60 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } },
  };

  return (
    <section
      className="py-24 px-6"
      style={{ backgroundColor: BG_DARK, color: TEXT_LIGHT }}
    >
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4" style={{ color: ACCENT }}>
            Our <span style={{ color: "#e94e20" }}>Powerful Features</span>
          </h2>
          <p className="text-lg max-w-3xl mx-auto" style={{ color: TEXT_FADED }}>
            Smart Posture uses <span style={{ color: ACCENT }}>cutting-edge technology</span> to monitor and improve your sitting habits
          </p>
        </motion.div>

        <motion.div
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={controls}
          className="grid grid-cols-1 md:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              className="p-8 rounded-xl cursor-pointer bg-[#251c17] 
                transition duration-300 hover:bg-[#c96442] hover:text-white hover:-translate-y-3"
              style={{ color: TEXT_LIGHT, border: `1.5px solid #c96442` }}
            >
              <div className="mb-5">{feature.icon}</div>
              <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
              <p style={{ color: TEXT_FADED }}>{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

const HowItWorksSection = () => {
  const steps = [
    {
      number: "01",
      title: "Enable Camera",
      description: <>Grant camera access to begin <span style={{ color: ACCENT }}>posture detection</span></>,
    },
    {
      number: "02",
      title: "Get Analyzed",
      description: <>Our ML model <span style={{ color: ACCENT }}>evaluates</span> your sitting position in real-time</>,
    },
    {
      number: "03",
      title: "View Results",
      description: <>Receive <span style={{ color: ACCENT }}>detailed feedback</span> on your posture and potential issues</>,
    },
    {
      number: "04",
      title: "Follow Recommendations",
      description: <>Practice <span style={{ color: ACCENT }}>suggested exercises</span> to correct and strengthen</>,
    },
  ];

  return (
    <section
      className="py-24 px-6"
      style={{ background: `linear-gradient(135deg, #2c2520, ${BG_DARK})`, color: TEXT_LIGHT }}
    >
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4" style={{ color: ACCENT }}>
            How <span style={{ color: "#e94e20" }}>It Works</span>
          </h2>
          <p className="text-lg max-w-3xl mx-auto" style={{ color: TEXT_FADED }}>
            Four simple steps to <span style={{ color: ACCENT }}>better posture</span> and improved well-being
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 60 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
              viewport={{ once: true }}
              className="relative px-6 py-4 border rounded-lg cursor-pointer bg-[#241b15] transition duration-300 hover:bg-[#c96442] hover:text-white hover:-translate-y-3"
              style={{ borderColor: "#c96442" }}
            >
              <div
                className="text-3xl font-extrabold rounded-full w-14 h-14 flex items-center justify-center mb-5"
                style={{ backgroundColor: ACCENT, color: TEXT_LIGHT }}
              >
                {step.number.slice(-1)}
              </div>
              <h3 className="text-lg font-bold mb-2">{step.title}</h3>
              <p style={{ color: TEXT_FADED }}>{step.description}</p>
              {index < steps.length - 1 && (
                <div
                  className="hidden md:block absolute top-8 left-full w-10 h-0.5"
                  style={{ backgroundColor: ACCENT, opacity: 0.3 }}
                />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

const BenefitsSection = () => {
  const benefits = [
    {
      metric: "70%",
      title: "Reduction in Back Pain",
      description: <>Users report <span style={{ color: ACCENT }}>significant decreases in discomfort</span> after 4 weeks</>,
    },
    {
      metric: "83%",
      title: "Improved Productivity",
      description: <>Better posture leads to <span style={{ color: ACCENT }}>better focus</span> and energy levels</>,
    },
    {
      metric: "2x",
      title: "Better Awareness",
      description: <>Users become <span style={{ color: ACCENT }}>twice as aware</span> of their posture habits</>,
    },
  ];

  return (
    <section
      className="py-24 px-6"
      style={{
        background: `linear-gradient(135deg, #3b271e, #622e20)`,
        color: TEXT_LIGHT,
      }}
    >
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4" style={{ color: ACCENT }}>
            Transform <span style={{ color: "#e94e20" }}>Your Health</span>
          </h2>
          <p className="text-lg max-w-3xl mx-auto" style={{ color: "#ecdbd1" }}>
            The benefits of <span style={{ color: ACCENT }}>good posture</span> extend beyond just sitting correctly
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {benefits.map((benefit, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: index * 0.2 }}
              viewport={{ once: true }}
              className="p-8 rounded-xl text-center bg-[#251c17] cursor-pointer transition duration-300 hover:bg-[#c96442] hover:text-white hover:-translate-y-3 border"
              style={{
                borderColor: "#c96442",
              }}
            >
              <div
                className="text-5xl font-extrabold mb-4"
                style={{
                  backgroundImage: `linear-gradient(90deg, #eabd8a, ${ACCENT})`,
                  WebkitBackgroundClip: "text",
                  color: "transparent",
                }}
              >
                {benefit.metric}
              </div>
              <h3 className="text-xl font-bold mb-3">{benefit.title}</h3>
              <p style={{ color: "#e0cfc6" }}>{benefit.description}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
          className="mt-16 flex justify-center"
        >
          <div
            className="flex items-center justify-center space-x-4 py-4 px-8 rounded-full"
            style={{
              backgroundColor: "rgba(255, 255, 255, 0.1)",
              color: "#f9edd5",
              fontWeight: "600",
            }}
          >
            <Award size={24} color="#ffdb57" />
            <span>Based on surveys from over <span style={{ color: ACCENT }}>10,000 Smart Posture users</span></span>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

const CTASection = () => {
  return (
    <section className="py-24 px-6" style={{ background: `#5B2D1F` }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="max-w-5xl mx-auto rounded-3xl overflow-hidden flex flex-col md:flex-row"
        style={{ background: `linear-gradient(125deg, ${ACCENT}, #b4563f)` }}
      >
        <div className="md:w-2/3 p-12 text-white flex flex-col justify-center">
          <h2 className="text-4xl font-extrabold mb-4" style={{ letterSpacing: "2px" }}>
            Ready to <span style={{ color: "#ffdb57" }}>improve your posture?</span>
          </h2>
          <p className="text-lg mb-8" style={{ color: "#ffd9c2" }}>
            Start your journey to better health and comfort with Smart Posture's advanced analysis
          </p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="inline-flex items-center px-8 py-4 rounded-lg font-bold text-white bg-black/20"
            style={{ letterSpacing: "1.2px" }}
          >
            Start Analysis <ArrowRight size={22} className="ml-3" />
          </motion.button>
        </div>
        <div className="md:w-1/3 p-8 bg-black/25 flex items-center justify-center">
          <PostureIllustration />
        </div>
      </motion.div>
    </section>
  );
};

const Footer = () => {
  return (
    <footer className="py-12 px-6" style={{ backgroundColor: BG_DARKER, color: TEXT_FADED }}>
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-10">
        <div>
          <h3 style={{ color: ACCENT, fontWeight: "700", fontSize: "1.4rem", marginBottom: "1rem" }}>
            Smart <span style={{ color: "#d65730" }}>Posture</span>
          </h3>
          <p>
            Improving health through <span style={{ color: ACCENT }}>better sitting habits</span>
          </p>
        </div>
        <div>
          <h4 style={{ color: "#d97b5d", fontWeight: "700", marginBottom: "1rem" }}>Features</h4>
          <ul style={{ lineHeight: 1.75 }}>
            <li><span style={{ color: ACCENT }}>Real-time Detection</span></li>
            <li>Posture Analysis</li>
            <li>Exercise Recommendations</li>
            <li>Progress Tracking</li>
          </ul>
        </div>
        <div>
          <h4 style={{ color: "#d97b5d", fontWeight: "700", marginBottom: "1rem" }}>Resources</h4>
          <ul style={{ lineHeight: 1.75 }}>
            <li>Blog</li>
            <li>Research</li>
            <li>Help Center</li>
            <li>Privacy Policy</li>
          </ul>
        </div>
        <div>
          <h4 style={{ color: "#d97b5d", fontWeight: "700", marginBottom: "1rem" }}>Contact</h4>
          <ul style={{ lineHeight: 1.75 }}>
            <li>Email: hello@smartposture.app</li>
            <li>Phone: (123) 456-7890</li>
            <li>Address: 123 Health St, Wellness City</li>
          </ul>
        </div>
      </div>
      <div style={{ borderTop: "1px solid #3d2920", marginTop: "3rem", paddingTop: "2rem", textAlign: "center", color: "#603e2f" }}>
        &copy; 2025  <span style={{ color: "#d65730" }}>Smart Posture</span>. All rights reserved.
      </div>
    </footer>
  );
};

const PostureIllustration = () => {
  return (
    <svg width="200" height="200" viewBox="0 0 280 240" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustration of correct posture">
      {/* Desk */}
      <rect x="40" y="140" width="200" height="10" fill="#c96442" />
      <rect x="50" y="150" width="180" height="60" fill="#b04a32" />
      {/* Computer */}
      <rect x="110" y="100" width="60" height="40" rx="2" fill="#973923" />
      <rect x="130" y="140" width="20" height="5" fill="#79321b" />
      {/* Chair */}
      <rect x="110" y="170" width="60" height="10" rx="2" fill="#6c2d16" />
      <rect x="130" y="180" width="20" height="40" fill="#582712" />
      {/* Head */}
      <circle cx="140" cy="80" r="15" fill="#f5dfd8" />
      {/* Spine */}
      <path d="M140 95 L140 130 L140 160" stroke="#f5dfd8" strokeWidth="6" strokeLinecap="round" />
      {/* Arms */}
      <path d="M140 110 L110 130 M140 110 L170 130" stroke="#f5dfd8" strokeWidth="4" strokeLinecap="round" />
      {/* Legs */}
      <path d="M140 160 L120 200 M140 160 L160 200" stroke="#f5dfd8" strokeWidth="4" strokeLinecap="round" />
      {/* Spine guideline */}
      <path d="M140 50 L140 220" stroke="#b04a32" strokeWidth="1" strokeDasharray="4 4" opacity="0.3" />
    </svg>
  );
};

export default Home;
