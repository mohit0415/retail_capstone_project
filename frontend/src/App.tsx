// src/App.tsx
// Role based routing (RBAC): every role gets /chat, the other screens are only
// opened when the backend (/auth/me) says the role has that screen.
// Miro board "Role to screen routing":
//   store_associate    -> chat
//   store_manager      -> chat, dashboard
//   compliance_officer -> chat, dashboard, review, slo
//   legal_reviewer     -> chat, review, slo
//   admin              -> chat, dashboard, review, slo, corpus

import { Navigate, Route, Routes } from 'react-router-dom'
import { useAuth } from './hooks/useAuth'
import Layout from './components/Layout'
import RequireAuth from './components/RequireAuth'
import RequireScreen from './components/RequireScreen'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'
import DashboardPage from './pages/DashboardPage'
import ReviewPage from './pages/ReviewPage'
import SloPage from './pages/SloPage'
import CorpusPage from './pages/CorpusPage'
import NotAllowedPage from './pages/NotAllowedPage'

function App() {
  const { isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="center-screen">
        <div className="row">
          <span className="spinner" /> <span className="muted">Loading Auth0...</span>
        </div>
      </div>
    )
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      <Route
        path="/"
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<ChatPage />} />
        <Route
          path="dashboard"
          element={
            <RequireScreen screen="dashboard">
              <DashboardPage />
            </RequireScreen>
          }
        />
        <Route
          path="review"
          element={
            <RequireScreen screen="review">
              <ReviewPage />
            </RequireScreen>
          }
        />
        <Route
          path="slo"
          element={
            <RequireScreen screen="slo">
              <SloPage />
            </RequireScreen>
          }
        />
        <Route
          path="corpus"
          element={
            <RequireScreen screen="corpus">
              <CorpusPage />
            </RequireScreen>
          }
        />
        <Route path="not-allowed" element={<NotAllowedPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  )
}

export default App
