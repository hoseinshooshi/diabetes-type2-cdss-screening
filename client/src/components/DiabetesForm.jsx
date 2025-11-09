import { useState } from "react";
import axios from "axios";

export default function EnhancedDiabetesForm({ setResult, setLoading }) {
  const [form, setForm] = useState({
    Sex: "female",
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    setLoading(true);
    
    try {
      const payload = {
        Sex: form.Sex,
        Pregnancies: form.Sex === "male" ? 0 : parseFloat(form.Pregnancies) || 0,
        Glucose: parseFloat(form.Glucose) || 0,
        BloodPressure: parseFloat(form.BloodPressure) || 0,
        SkinThickness: parseFloat(form.SkinThickness) || 29,
        Insulin: parseFloat(form.Insulin) || 80,
        BMI: parseFloat(form.BMI) || 0,
        DiabetesPedigreeFunction: parseFloat(form.DiabetesPedigreeFunction) || 0.5,
        Age: parseFloat(form.Age) || 0
      };

      const response = await axios.post("http://localhost:5000/predict", payload);
      setResult(response.data);
      
    } catch (error) {
      console.error("Error:", error);
      alert("Server connection error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center pt-40" id="form">
    <div className=" overflow-hidden border-4 border-blue p-6 shadow-lg w-5/6 max-w-2xl">
      <h2 className="text-2xl font-bold text-center mb-4 text-gray-50">
        📝 Health Assessment Form
      </h2>
      
      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Gender */}
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-white mb-2">
            Gender *
          </label>
          <select
            name="Sex"
            value={form.Sex}
            onChange={handleChange}
            className="w-full bg-blue placeholder-opaque-black font-semibold p-3 mt-5 text-black font-playfair"
          >
            <option value="female">Female</option>
            <option value="male">Male</option>
          </select>
        </div>

        {/* Pregnancies */}
        {form.Sex === "female" && (
          <div>
            <label className="block text-sm font-medium text-white mb-2">
              Pregnancies
            </label>
            <input
              type="number"
              name="Pregnancies"
              value={form.Pregnancies}
              onChange={handleChange}
              className="w-full bg-blue placeholder-opaque-black font-semibold p-3 mt-5 text-black font-playfair"
              placeholder="Example: 2"
              min="0"
            />
          </div>
        )}

        {/* Other fields */}
        {[
          { name: "Age", label: "Age (years)", placeholder: "Example: 35" },
          { name: "Glucose", label: "Glucose (mg/dL)", placeholder: "Example: 120" },
          { name: "BloodPressure", label: "Blood Pressure (mmHg)", placeholder: "Example: 80" },
          { name: "BMI", label: "BMI", placeholder: "Example: 26.5", step: "0.1" },
          { name: "SkinThickness", label: "Skin Thickness (mm)", placeholder: "Example: 25" },
          { name: "Insulin", label: "Insulin (mu U/ml)", placeholder: "Example: 100" },
          { name: "DiabetesPedigreeFunction", label: "Diabetes Pedigree", placeholder: "Example: 0.5", step: "0.001" },
        ].map(field => (
          <div key={field.name}>
            <label className="block text-sm font-medium text-white mb-2">
              {field.label}
            </label>
            <input
              type="number"
              name={field.name}
              value={form[field.name]}
              onChange={handleChange}
              step={field.step || "1"}
              className="w-full bg-blue placeholder-opaque-black font-semibold p-3 mt-5 text-black font-playfair"
              placeholder={field.placeholder}
            />
          </div>
        ))}

        <div className="md:col-span-2 mt-4 flex w-full items-center justify-center">
          <button
            type="submit"
            className="p-5 bg-yellow font-semibold text-deep-blue mt-5 hover:bg-red
                        hover:text-white transition duration-500">
            🔍 Analyze Results
          </button>
        </div>
      </form>
    </div>
    </div>
  );
}