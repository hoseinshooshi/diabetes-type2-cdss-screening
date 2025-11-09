export default function EnhancedResultView({ result, setResult, setLoading }) {
  const handleNewTest = () => {
    setLoading(false);
    setResult(null);
  };

  return (
    <div className="flex items-center justify-center pt-40">
    <div className="bg-white px-6 rounded-2xl shadow-lg max-w-4xl w-4/6">
      {/* Result Header */}
      <div className={`text-center p-6 rounded-xl mb-6 mt-4 ${
        result.prediction === 1 
          ? 'bg-red-50 border border-red-200' 
          : 'bg-green-50 border border-green-200'
      }`}>
        <h2 className={`text-3xl font-bold mb-3 ${
          result.prediction === 1 ? 'text-red-700' : 'text-green-700'
        }`}>
          {result.prediction === 1 ? "High Diabetes Risk" : "Normal - No Diabetes"}
        </h2>
        <div className={`inline-block px-4 py-2 rounded-full text-sm font-medium ${
          result.risk_level === 'High' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
        }`}>
          Risk Level: {result.risk_level}
        </div>
      </div>

      {/* Recommendations */}
      {result.prediction === 1 ? (
        <div className="mb-6">
          <h3 className="font-bold text-red-700 mb-4 text-lg">Medical Recommendations</h3>
          <div className="space-y-3">
            {result.recommendations.map((rec, index) => (
              <div key={index} className="p-4 border border-red-200 bg-red-50 rounded-lg">
                <div className="font-bold text-red-800">{rec.category}:</div>
                <div className="text-red-700">{rec.advice}</div>
              </div>
            ))}
            {result.ai_analysis && (
              <div className="mt-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
                <h3 className="font-bold text-purple-800 mb-3 flex items-center">
                  🧠 تحلیل هوشمند Gemini
                  <span className="text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded-full mr-2">
                    AI
                  </span>
                </h3>
                <div className="text-purple-700 whitespace-pre-line">
                  {result.ai_analysis.analysis}
                </div>
                <div className="mt-2 text-xs text-purple-500">
                  منبع: {result.ai_analysis.source}
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <h4 className="font-bold text-red-800 mb-2">🚨 Urgent Actions</h4>
            <p className="text-red-700">{result.followup}</p>
          </div>
        </div>
      ) : (
        <div className="mb-6">
          <h3 className="font-bold text-green-700 mb-4 text-lg">Preventive Recommendations</h3>
          <div className="space-y-3 mb-4">
            {result.recommendations.map((rec, index) => (
              <div key={index} className="p-4 border border-green-200 bg-green-50 rounded-lg">
                <div className="font-bold text-green-800">{rec.category}:</div>
                <div className="text-green-700">{rec.advice}</div>
              </div>
            ))}
          </div>
          
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <h4 className="font-bold text-green-800 mb-2">📅 Next Screening</h4>
            <p className="text-green-700">Next screening: {result.next_screening}</p>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center pb-4">
        <button
          onClick={handleNewTest}
          className="p-5 bg-yellow font-semibold text-deep-blue mt-5 hover:bg-red
                        hover:text-white transition duration-500">
          🔄 New Test
        </button>
      </div>
    </div>
    </div>
  );
}