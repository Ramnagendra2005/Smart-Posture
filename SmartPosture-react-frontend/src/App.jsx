import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useAuth } from "./AuthContext"; // Custom hook from AuthContext.js
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import Analysis from "./components/Analysis";
import Report from "./components/Report";
import ChatBot from "./components/Chatbot";
import SignInButton from "./components/SignInButton";

// ProtectedRoute wraps each route to check if the user is logged in.
// It uses the useAuth hook to grab the user and loading states, and returns the SignInButton if user is not available.
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) return <div>Loading...</div>;
  if (!user)
    return (
      <div style={{ margin: "2rem" }}>
        <SignInButton />
      </div>
    );
  return children;
};


function App() {
  return (
    <Router>
      <ProtectedRoute>
          <Navbar />
      </ProtectedRoute>
      <Routes>
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Home />
            </ProtectedRoute>
          }
        />
        <Route
          path="/analysis"
          element={
            <ProtectedRoute>
              <Analysis />
            </ProtectedRoute>
          }
        />
        <Route
          path="/report"
          element={
            <ProtectedRoute>
              <Report />
            </ProtectedRoute>
          }
        />
      </Routes>
      <ChatBot />
    </Router>
  );
}

export default App;

