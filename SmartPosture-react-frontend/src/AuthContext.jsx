// src/AuthContext.js
import React, { createContext, useContext, useState, useEffect } from "react";
import { motion } from "framer-motion";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/user", {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          setUser(data);
          console.log(data);
          console.log(data.picture);
        }
      } catch (error) {
        console.error("Error fetching user:", error);
      }
      setLoading(false);
    };

    fetchUser();
  }, []);

  const login = () => {
    window.location.href = "http://localhost:5000/login";
  };

  // Modified SignInButton using a similar animated template as in the first code example
  // but using "Sign in with Google" as in the second code.
  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading,  logout,login }}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook for easy access to the AuthContext
export const useAuth = () => {
  return useContext(AuthContext);
};

export default AuthContext;
