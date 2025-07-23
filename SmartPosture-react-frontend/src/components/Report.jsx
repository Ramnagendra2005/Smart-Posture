import { useState, useEffect, useRef } from 'react';

export default function Report() {
  // Color palette (matching Analysis.jsx)
  const colors = {
    bgGradient: "from-[#171413] to-[#1f1a18]",
    cardBg: "bg-[#251c17]",
    headingGradient: "text-[#C96442]",
    accent: "#C96442",
    text: "text-[#eee]",
    textSecondary: "text-[#aaa]",
    success: "text-green-400",
    warning: "text-[#e94e20]",
    error: "text-red-500",
  };

  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [graphsVisible, setGraphsVisible] = useState({});
  const [graphLoadStates, setGraphLoadStates] = useState({
    postureOverTime: { loading: true, error: false },
    postureComponents: { loading: true, error: false },
    blinkRate: { loading: true, error: false },
    sessionComparison: { loading: true, error: false }
  });

  const cacheBuster = useRef(Date.now());
  const graphRefs = {
    postureOverTime: useRef(null),
    postureComponents: useRef(null),
    blinkRate: useRef(null),
    sessionComparison: useRef(null)
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:5000/report/data?t=${cacheBuster.current}`);
        const data = await response.json();
        setReportData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching report data:', error);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (loading) return;
    const observerOptions = {
      root: null,
      rootMargin: '0px',
      threshold: 0.3
    };

    const observerCallback = (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setGraphsVisible(prev => ({
            ...prev,
            [entry.target.id]: true
          }));
        }
      });
    };

    const observer = new window.IntersectionObserver(observerCallback, observerOptions);

    Object.entries(graphRefs).forEach(([key, ref]) => {
      if (ref.current) observer.observe(ref.current);
    });

    return () => observer.disconnect();
  }, [loading, graphRefs]);

  const handleImageLoad = (graphName) => {
    setGraphLoadStates(prev => ({
      ...prev,
      [graphName]: { ...prev[graphName], loading: false }
    }));
  };

  const handleImageError = (graphName) => {
    setGraphLoadStates(prev => ({
      ...prev,
      [graphName]: { loading: false, error: true }
    }));
  };

  // Score ring with warm accent and smooth animation
  const ScoreRing = ({ score, size = 180, strokeWidth = 8 }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (score / 100) * circumference;

    let color = '#ef4444';
    if (score >= 80) color = '#10b981';
    else if (score >= 60) color = '#f59e0b';

    return (
      <div className="relative flex flex-col items-center justify-center">
        <svg height={size} width={size} className="progress-ring">
          <circle
            className="text-gray-700"
            stroke="currentColor"
            fill="transparent"
            strokeWidth={strokeWidth}
            r={radius}
            cx={size / 2}
            cy={size / 2}
          />
          <circle
            className="progress-ring__circle"
            stroke={color}
            fill="transparent"
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            r={radius}
            cx={size / 2}
            cy={size / 2}
            style={{ "--offset": strokeDashoffset }}
          />
        </svg>
        <div className="absolute flex flex-col items-center justify-center text-center">
          <span className="text-4xl font-bold text-[#C96442]">{score}</span>
          <span className="text-sm text-[#aaa]">out of 100</span>
        </div>
      </div>
    );
  };

  // Stat card with accent border and spacing matching Analysis page
  const StatCard = ({ icon, title, value, trend }) => {
    let trendColor = "text-[#aaa]";
    let trendIcon = "→";

    if (trend === "improving") {
      trendColor = "text-green-400";
      trendIcon = "↑";
    } else if (trend === "declining") {
      trendColor = "text-red-500";
      trendIcon = "↓";
    }

    return (
      <div className={`${colors.cardBg} rounded-xl p-4 flex flex-col stats-container border border-[rgba(201,100,66,0.2)]`}>
        <div className="flex items-center mb-2">
          <div className="text-[#C96442] mr-2 text-lg">{icon}</div>
          <h3 className={`${colors.textSecondary} text-sm`}>{title}</h3>
        </div>
        <div className="flex items-end justify-between">
          <div className={`${colors.text} text-2xl font-bold`}>{value}</div>
          {trend && (
            <div className={`${trendColor} text-sm font-medium`}>
              {trendIcon}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Component score styled with accent colors
  const ComponentScore = ({ title, score, icon }) => {
    let color = 'bg-red-500';
    if (score >= 80) color = 'bg-green-500';
    else if (score >= 60) color = 'bg-yellow-500';

    return (
      <div className="flex flex-col items-center text-center">
        <div className={`w-12 h-12 ${color} rounded-full flex items-center justify-center mb-2`}>
          {icon}
        </div>
        <div className={`${colors.textSecondary} text-sm font-medium mb-1`}>{title}</div>
        <div className={`${colors.text} text-lg font-bold`}>{score}</div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#171413] to-[#1f1a18]">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-[#C96442] border-t-transparent mb-4"></div>
          <p className={`${colors.textSecondary}`}>Loading your posture report...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen text-white bg-gradient-to-br ${colors.bgGradient}`} style={{ marginTop: "100px" }}>
      {/* Header */}
      <header className="py-6 px-8 bg-gradient-to-r from-[#C96442] to-[#b4563f] shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:justify-between md:items-center">
          <h1 className="text-3xl font-bold tracking-wide">
            <span className={`bg-clip-text text-transparent bg-gradient-to-r from-[#C96442] to-[#b4563f]`}>Posture Report</span>
          </h1>
          <p className={`${colors.textSecondary} mt-1 md:mt-0`}>An analysis of your posture health</p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-12">
        {/* Summary Section */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className={`${colors.cardBg} rounded-xl p-6 flex flex-col items-center shadow-lg hover:shadow-xl transition-shadow duration-300`}>
            <h2 className={`text-xl font-semibold mb-4 ${colors.textSecondary}`}>Overall Posture Score</h2>
            <ScoreRing score={reportData?.currentScore || 0} />
            <p className={`mt-4 text-center ${colors.textSecondary} text-sm`}>
              {reportData?.scoreTrend === "improving" && "Your posture is improving! Keep up the good work."}
              {reportData?.scoreTrend === "declining" && "Your posture has been declining. Try to be more mindful."}
              {reportData?.scoreTrend === "stable" && "Your posture has been stable recently."}
            </p>
          </div>

          <div className={`${colors.cardBg} rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300`}>
            <h2 className={`text-xl font-semibold mb-4 ${colors.textSecondary}`}>Today's Summary</h2>
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                icon="⏱️"
                title="Time Tracked"
                value={`${reportData?.timeTracked || 0} hrs`}
              />
              <StatCard
                icon="🔄"
                title="Total Sessions"
                value={reportData?.totalSessions || 0}
              />
              <StatCard
                icon="⚠️"
                title="Corrections"
                value={reportData?.totalCorrections || 0}
                trend={reportData?.scoreTrend}
              />
              <StatCard
                icon="☕"
                title="Breaks Taken"
                value={reportData?.totalBreaks || 0}
              />
            </div>
          </div>

          <div className={`${colors.cardBg} rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300`}>
            <h2 className={`text-xl font-semibold mb-4 ${colors.textSecondary}`}>Component Analysis</h2>
            <div className="grid grid-cols-3 gap-4">
              <ComponentScore
                title="Head"
                score={reportData?.headTiltScore || 0}
                icon="🧠"
              />
              <ComponentScore
                title="Shoulders"
                score={reportData?.shoulderAlignmentScore || 0}
                icon="💪"
              />
              <ComponentScore
                title="Spine"
                score={reportData?.spinalPostureScore || 0}
                icon="🦴"
              />
              <ComponentScore
                title="Hips"
                score={reportData?.hipBalanceScore || 0}
                icon="🦯"
              />
              <ComponentScore
                title="Legs"
                score={reportData?.legPositionScore || 0}
                icon="🦵"
              />
            </div>
          </div>
        </section>

        {/* Weekly Overview */}
        <section className="space-y-6">
          <h2 className={`text-2xl font-bold ${colors.headingGradient}`}>Weekly Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className={`${colors.cardBg} rounded-xl p-6 shadow-md`}>
              <h3 className={`text-lg font-semibold mb-2 ${colors.textSecondary}`}>Your Average Score</h3>
              <div className="flex items-center gap-6">
                <div className="text-4xl font-bold text-[#C96442]">{reportData?.averageScore || 0}</div>
                <div className={`text-sm ${colors.textSecondary}`}>
                  {reportData?.averageScore > 80 ?
                    <span>Great job maintaining excellent posture!</span> :
                    <span>There's room for improvement in your posture habits.</span>}
                </div>
              </div>
            </div>

            <div className={`${colors.cardBg} rounded-xl p-6 shadow-md`}>
              <h3 className={`text-lg font-semibold mb-2 ${colors.textSecondary}`}>Weekly Progress</h3>
              <div className="flex justify-between items-center">
                <div className={`${colors.textSecondary} text-sm space-y-1`}>
                  <div>Sessions: <span className="font-medium">{reportData?.totalSessions || 0}</span></div>
                  <div>Time Tracked: <span className="font-medium">{reportData?.timeTracked || 0} hrs</span></div>
                  <div>Corrections: <span className="font-medium">{reportData?.totalCorrections || 0}</span></div>
                </div>
                <div className="text-5xl font-light text-[#C96442] select-none">
                  {reportData?.scoreTrend === "improving" ? "↗" : reportData?.scoreTrend === "declining" ? "↘" : "→"}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Graphs Section */}
        <section>
          <h2 className={`text-2xl font-bold mb-6 ${colors.headingGradient}`}>Detailed Analysis</h2>
          {['postureOverTime', 'postureComponents', 'blinkRate', 'sessionComparison'].map((graphName, idx) => {
            const titles = {
              postureOverTime: "Posture Score Trend",
              postureComponents: "Posture Component Analysis",
              blinkRate: "Blink Rate Analysis",
              sessionComparison: "Session Comparison"
            };
            const descriptions = {
              postureOverTime: "Your posture score has been " + (reportData?.scoreTrend || "stable") + " over the last week.",
              postureComponents: "Comparison of posture components between today and yesterday.",
              blinkRate: "Your blink rate throughout the day compared to the recommended rate of 15 blinks per minute.",
              sessionComparison: "Comparison of posture scores and correction count across your recent sessions."
            };

            return (
              <div
                key={graphName}
                id={graphName}
                ref={graphRefs[graphName]}
                className={`${colors.cardBg} rounded-xl p-6 mb-10 shadow-md transition-all duration-700 ease-out ${graphsVisible[graphName] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
                style={{ transitionDelay: `${idx * 0.15}s` }}
              >
                <h3 className={`text-xl font-semibold mb-4 ${colors.textSecondary}`}>{titles[graphName]}</h3>
                <div className="relative rounded-lg overflow-hidden h-64 bg-[rgba(201,100,66,0.1)] flex items-center justify-center">
                  {graphLoadStates[graphName].loading && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className={`animate-spin rounded-full h-12 w-12 border-4 border-[#C96442] border-t-transparent`}></div>
                    </div>
                  )}
                  {graphLoadStates[graphName].error ? (
                    <div className={`absolute inset-0 flex items-center justify-center text-red-500 p-4 text-center ${colors.text}`}>
                      Failed to load graph. Please try refreshing.
                    </div>
                  ) : (
                    <img
                      src={`http://localhost:5000/graph/${graphName.replace(/([A-Z])/g, '_$1').toLowerCase()}?t=${cacheBuster.current}`}
                      alt={titles[graphName]}
                      className="w-full h-full object-contain"
                      onLoad={() => handleImageLoad(graphName)}
                      onError={() => handleImageError(graphName)}
                    />
                  )}
                </div>
                <p className={`mt-4 text-sm ${colors.textSecondary}`}>
                  {descriptions[graphName]}
                </p>
              </div>
            );
          })}
        </section>

        {/* Recommendations Section */}
        <section>
          <h2 className={`text-2xl font-bold mb-6 ${colors.headingGradient}`}>Recommendations</h2>
          <div className={`${colors.cardBg} rounded-xl p-6 shadow-md space-y-6`}>
            <div>
              <h3 className={`text-lg font-semibold mb-2 ${colors.textSecondary}`}>Improve Your Posture</h3>
              <ul className={`${colors.textSecondary} list-disc pl-5 space-y-2`}>
                <li>Take regular breaks - stand up every 30 minutes</li>
                <li>Position your monitor at eye level to avoid neck strain</li>
                <li>Keep your shoulders relaxed and pulled back slightly</li>
                <li>Use a chair with proper lumbar support</li>
                <li>Practice core-strengthening exercises to support your spine</li>
              </ul>
            </div>
            <div>
              <h3 className={`text-lg font-semibold mb-2 ${colors.textSecondary}`}>Eye Health Recommendations</h3>
              <ul className={`${colors.textSecondary} list-disc pl-5 space-y-2`}>
                <li>Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds</li>
                <li>Consciously blink more frequently to prevent dry eyes</li>
                <li>Adjust screen brightness to match your surroundings</li>
                <li>Consider using blue light filtering glasses for extended screen time</li>
              </ul>
            </div>
          </div>
        </section>
      </main>

      <footer className={`${colors.cardBg} py-6 mt-12`}>
        <div className="max-w-7xl mx-auto px-6 text-center text-[#aaa] text-sm select-none">
          © 2025 Posture Monitor | Your digital posture assistant
        </div>
      </footer>

      {/* Global styles and progress ring animation */}
      <style jsx global>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes ring-progress {
          from { stroke-dashoffset: 1000; }
          to { stroke-dashoffset: var(--offset); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        .progress-ring__circle {
          animation: ring-progress 1.5s ease-out forwards;
          transform: rotate(-90deg);
          transform-origin: 50% 50%;
        }
      `}</style>
    </div>
  );
}
