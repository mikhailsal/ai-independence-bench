import { Routes, Route } from 'react-router-dom'
import { ManifestProvider } from './lib/manifest'
import Layout from './components/Layout'
import Leaderboard from './pages/Leaderboard'
import ModelDetail from './pages/ModelDetail'
import TrajectoryViewer from './pages/TrajectoryViewer'
import ScenarioExplorer from './pages/ScenarioExplorer'
import About from './pages/About'

export default function App() {
  return (
    <ManifestProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/model/:modelId" element={<ModelDetail />} />
          <Route path="/trajectory/:modelId/:run/:scenario" element={<TrajectoryViewer />} />
          <Route path="/explore/:scenarioId?" element={<ScenarioExplorer />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </ManifestProvider>
  )
}
