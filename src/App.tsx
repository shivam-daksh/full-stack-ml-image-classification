import React, { useRef, useState } from "react";
import { Upload, Loader2 } from "lucide-react";

interface Prediction {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
}

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setPredictions([]);

    // Create FormData
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Send to backend
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process image");
      }

      const data = await response.json();
      setImageUrl(data.processed_image);
      setPredictions(data.predictions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setImageUrl(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadClick = () => {
    inputRef.current?.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-8 text-center">
            Image Classification
          </h1>

          <div className="space-y-8">
            <div
              onClick={handleUploadClick}
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-500 transition-colors"
            >
              <input
                type="file"
                ref={inputRef}
                className="hidden"
                accept="image/*"
                onChange={handleImageUpload}
              />
              {!imageUrl ? (
                <div className="space-y-4">
                  <Upload className="w-12 h-12 mx-auto text-gray-400" />
                  <p className="text-gray-500">
                    Click to upload an image or drag and drop
                  </p>
                </div>
              ) : (
                <div className="relative">
                  <img
                    src={imageUrl}
                    alt="Uploaded"
                    className="max-h-96 mx-auto rounded-lg"
                  />
                  <div className="text-center mt-4">
                    <a
                      href={imageUrl}
                      download="processed_image.png"
                      className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      Save
                    </a>
                  </div>
                </div>
              )}
            </div>

            {isLoading && (
              <div className="flex items-center justify-center space-x-2 text-indigo-600">
                <Loader2 className="w-6 h-6 animate-spin" />
                <span>Processing image...</span>
              </div>
            )}

            {error && (
              <div className="bg-red-50 text-red-700 p-4 rounded-lg">
                {error}
              </div>
            )}

            {predictions.length > 0 && !isLoading && (
              <div className="bg-gray-50 rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">
                  Predictions
                </h2>
                <div className="space-y-3">
                  {predictions.map((prediction, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-700">
                          {prediction.class_name}
                        </span>
                        <span className="text-gray-500">
                          {(prediction.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      {/* <div className="text-gray-500 text-sm">
                        BBox: [{prediction.bbox.join(", ")}]
                      </div> */}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
