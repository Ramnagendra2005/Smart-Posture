import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../AuthContext";

// Font import (put this in index.html or use a head-manager in React root if needed):
// <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap" rel="stylesheet">

const ACCENT = "#C96442";
const BG_DARK = "#171413";
const BG_DARKER = "#120f0e";
const WHITE = "#fff";
const LIGHTER = "#f8e9e4";
const FADED = "#d1cec9";

const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { user, logout } = useAuth();
  const location = useLocation();
  const [profileImageError, setProfileImageError] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => setProfileImageError(false), [user]);

  const getUserDisplayName = () => {
    if (!user) return "User";
    return (
      user.username ||
      user.name ||
      user.firstName ||
      user.displayName ||
      user.email?.split("@")[0] ||
      "User"
    );
  };

  const getUserProfileImage = () => {
    if (!user || profileImageError) return "/default-avatar.png";
    const imageUrl = user.profilePicture || user.picture || user.avatar || user.imageUrl;
    if (!imageUrl) return "/default-avatar.png";
    const isValidUrl = imageUrl.startsWith("http") || imageUrl.startsWith("/");
    return isValidUrl ? imageUrl : "/default-avatar.png";
  };

  const handleImageError = () => setProfileImageError(true);

  return (
    <>
      <style>{`
        * { font-family: 'Inter', system-ui, sans-serif; }
        .nav-blur { /* reserved if blur wanted in future */ }
        .logo-container {
          display: flex;
          align-items: center;
          gap: 6px;
          user-select: none;
          cursor: pointer;
        }
        .logo-brand {
          font-weight: 900;
          letter-spacing: 1px;
          font-size: 1.5rem;
          color: ${ACCENT};
          text-shadow: 0 2px 14px ${ACCENT}20;
          transition: color 0.18s, text-shadow 0.18s;
        }
        .logo-container:hover .logo-brand {
          color: #e37251;
          text-shadow: 0 4px 26px ${ACCENT}44;
        }
        /* Nav Links */
        .nav-link {
          position: relative;
          font-size: 1rem;
          font-weight: 600;
          color: ${WHITE};
          padding: 0.6em 1em;
          border-radius: 10px;
          transition: 
            color 0.18s,
            background 0.16s,
            box-shadow 0.22s;
        }
        .nav-link:not(.active):hover,
        .nav-link:not(.active):focus {
          color: ${ACCENT};
          background: #251a17;
          text-shadow: 0 2px 6px ${ACCENT}30;
        }
        .nav-link.active {
          color: ${ACCENT};
          background: #251a1a;
          box-shadow: 0 2px 14px ${ACCENT}33;
        }
        .nav-link::after {
          content: "";
          position: absolute;
          left: 1em; right: 1em; bottom: .34em;
          height: 2px;
          background: linear-gradient(90deg, ${ACCENT}, transparent 100%);
          border-radius: 1px;
          opacity: 0;
          transform: scaleX(0.5);
          transition: opacity 0.2s, transform 0.22s;
        }
        .nav-link.active::after,
        .nav-link:hover::after,
        .nav-link:focus::after {
          opacity: 1;
          transform: scaleX(1);
        }
        /* Auth Buttons */
        .btn-accent {
          background: ${ACCENT};
          color: ${WHITE} !important;
          box-shadow: 0 2px 10px ${ACCENT}22, 0 1.5px 0 #0002;
          font-size: 1rem;
          font-weight: 700;
          border-radius: 11px;
          border: none;
          padding: 0.57em 1.25em;
          transition: background 0.15s, box-shadow 0.17s, color 0.18s;
        }
        .btn-accent:hover, .btn-accent:focus {
          background: #e37251;
          color: ${WHITE} !important;
          box-shadow: 0 4px 18px ${ACCENT}4a;
        }
        .btn-outline-accent {
          background: none;
          border: 2px solid ${ACCENT};
          color: ${ACCENT};
          font-weight: 600;
          border-radius: 10px;
          padding: 0.54em 1.18em;
          font-size: 1rem;
          transition: 
            background 0.15s, color 0.17s, border-color 0.18s;
        }
        .btn-outline-accent:hover, .btn-outline-accent:focus {
          background: ${ACCENT};
          color: ${WHITE};
          border-color: #e0633e;
        }
        /* Avatar animation */
        .avatar {
          background: #201a17;
          border: 2.5px solid #31251d;
          box-shadow: 0 0 0px #000;
          width: 36px; height: 36px;
          border-radius: 40px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: border 0.18s, box-shadow 0.17s;
        }
        .avatar img {
          width: 100%; height: 100%; object-fit: cover;
          display: block;
        }
        .avatar:hover, .avatar:focus {
          border-color: ${ACCENT};
          box-shadow: 0 0 12px ${ACCENT}88;
        }
        /* Transitions for Mobile */
        .mobile-menu {
          background: ${BG_DARKER};
          min-height: 100vh;
          animation: fadeInMobile 0.13s;
        }
        @keyframes fadeInMobile {
          from { opacity: 0; transform: translateY(30px);}
          to   { opacity: 1; transform: translateY(0);}
        }
        .mobile-item {
          opacity: 0;
          animation: fadeInMobile 0.36s cubic-bezier(.4,0,.2,1) .06s forwards;
        }
        .mobile-item + .mobile-item { animation-delay: .13s; }
        .mobile-auth { animation-delay: .28s !important; }
        /* Fancy Close Button */
        .close-btn svg {
          transition: color 0.2s;
        }
        .close-btn:hover svg, .close-btn:focus svg {
          color: ${ACCENT};
        }
      `}</style>
      <nav
        style={{
          background: BG_DARK,
          borderBottom: scrolled ? `1.25px solid #382018` : "none",
          boxShadow: scrolled
            ? `0 8px 16px -10px #b2452a14`
            : "none",
        }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ease-out`}
      >
        <div className={`max-w-7xl mx-auto px-4 md:px-8 flex items-center justify-between transition-all duration-300 ${scrolled ? "py-3" : "py-5"}`}>
          {/* Logo */}
          <Link to="/" className="logo-container">
            <span className="logo-brand">SmartPosture</span>
          </Link>
          {/* Desktop Navigation */}
          <div className="hidden md:flex gap-6 m-auto">
            {[
              { path: "/", label: "Home" },
              { path: "/analysis", label: "Analysis" },
              { path: "/report", label: "Report" },
            ].map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                label={item.label}
                active={location.pathname === item.path}
              />
            ))}
          </div>
          {/* Auth Section - Desktop */}
          <div className="hidden md:block">
            {user ? (
              <div className="flex items-center gap-3">
                <div className="avatar" tabIndex={0}>
                  <img
                    src={getUserProfileImage()}
                    alt="Profile"
                    onError={handleImageError}
                  />
                </div>
                <span 
                  className="text-white"
                  style={{
                    fontWeight: 500,
                    fontSize: "1rem",
                    letterSpacing: ".01em",
                  }}
                >
                  {getUserDisplayName()}
                </span>
                <button
                  onClick={logout}
                  className="btn-outline-accent"
                  style={{ marginLeft: ".2em" }}
                >Sign out</button>
              </div>
            ) : (
              <Link to="/signin" className="btn-accent">Sign In</Link>
            )}
          </div>
          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-2xl hover:bg-[#221b18] transition-all duration-200 close-btn"
            aria-label="Toggle menu"
          >
            <svg
              className="w-7 h-7"
              style={{ color: WHITE }}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {mobileMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"/>
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 6h16M4 12h16M4 18h16"/>
              )}
            </svg>
          </button>
        </div>
        {/* Mobile Menu Overlay */}
        <div
          className={`fixed inset-0 mobile-menu flex flex-col items-center justify-center transition-all z-50
          ${mobileMenuOpen ? "opacity-100 visible" : "opacity-0 invisible pointer-events-none"}
          md:hidden`}
        >
          <div className="flex flex-col items-center  space-y-7 mb-10">
            {[
              { path: "/", label: "Home" },
              { path: "/analysis", label: "Analysis" },
              { path: "/report", label: "Report" },
            ].map((item, idx) => (
              <Link
                key={item.label}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`mobile-item text-2xl font-extrabold relative`}
                style={{
                  color: location.pathname === item.path ? ACCENT : WHITE,
                  letterSpacing: ".03em",
                  textShadow: location.pathname === item.path
                    ? `0 4px 20px ${ACCENT}44`
                    : `0 2px 10px #0009`,
                }}
              >
                <span
                  style={{
                    position: "relative",
                    paddingBottom: "0.08em",
                    borderBottom: location.pathname === item.path
                      ? `2.5px solid ${ACCENT}`
                      : `2.5px solid transparent`,
                    transition: "border .17s",
                  }}
                >
                  {item.label}
                </span>
              </Link>
            ))}
          </div>
          <div className="mobile-item mobile-auth">
            {user ? (
              <div className="flex flex-col items-center space-y-4">
                <div className="avatar" tabIndex={0} style={{ width: 52, height: 52 }}>
                  <img
                    src={getUserProfileImage()}
                    alt="Profile"
                    onError={handleImageError}
                  />
                </div>
                <span className="text-white font-semibold text-lg" style={{ letterSpacing: ".014em" }}>
                  {getUserDisplayName()}
                </span>
                <button
                  onClick={() => {
                    logout();
                    setMobileMenuOpen(false);
                  }}
                  className="btn-outline-accent"
                  style={{ width: 140, marginTop: ".3em" }}
                >Sign out</button>
              </div>
            ) : (
              <Link
                to="/signin"
                onClick={() => setMobileMenuOpen(false)}
                className="btn-accent"
                style={{ width: 150 }}
              >
                Sign In
              </Link>
            )}
          </div>
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="absolute top-4 right-4 p-2 rounded-lg hover:bg-[#221b18] close-btn"
            aria-label="Close menu"
          >
            <svg className="w-6 h-6" style={{ color: WHITE }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </nav>
    </>
  );
};

const NavLink = ({ to, label, active }) => (
  <Link
    to={to}
    className={`nav-link${active ? " active" : ""}`}
    tabIndex={0}
    aria-current={active ? "page" : undefined}
  >
    {label}
  </Link>
);

export default Navbar;
