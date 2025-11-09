import { useEffect, useState } from 'react'
import useMediaQuery from './hooks/useMediaQuery'
import Navbar from './components/Navbar';
import ContactMe from './components/ContactMe';
import Footer from './components/Footer';
import EnhancedDiabetesForm from './components/DiabetesForm';
import EnhancedResultView from './components/ResultRev';
function App() {
  const [selectedPage, setSelectedPage] = useState('form');
  const isAboveMediumScreen = useMediaQuery("(min-width: 1060px)")
  const [isTopofPage , setISTopOfPage] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    const handleScroll = () => {
      if(window.scrollY === 0) setISTopOfPage(true);
      if (window.scrollY !== 0) setISTopOfPage(false);
    }
    window.addEventListener("scroll", handleScroll);
    return() => window.removeEventListener("scroll", handleScroll)
  }, [])
  return (
    <div className="app bg-deep-blue ">
      <Navbar selectedPage={selectedPage} setSelectedPage={setSelectedPage} isTopofPage={isTopofPage} />
      <main className="w-full max-w-6xl flex-1">
        {loading ? (
          <div className="flex flex-col items-center justify-center pt-20 h-[100vh] w-full">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mb-4"></div>
            <p className="text-gray-600 text-lg">Analyzing...</p>
          </div>
        ) : !result ? (
          <EnhancedDiabetesForm 
            setResult={setResult} 
            setLoading={setLoading}
          />
        ) : (
          <EnhancedResultView 
            result={result} 
            setResult={setResult}
            setLoading={setLoading}
          />
        )}
      </main>
      <div className='w-5/6 md:h-full mx-auto mt-48'>
        <ContactMe />
      </div>
      <div className='mt-32'>
        <Footer />
      </div>
    </div>
  )
}
export default App
