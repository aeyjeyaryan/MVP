import React, { useEffect, useState, useCallback } from 'react';
import { Camera, RefreshCw } from 'lucide-react';

function App() {
  const [videoFeed, setVideoFeed] = useState<string>('');
  const [prediction, setPrediction] = useState<string>('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnecting, setIsConnecting] = useState(true);

  const connectWebSocket = useCallback(() => {
    setIsConnecting(true);
    // Update WebSocket URL to connect to the FastAPI backend on port 8000
    const wsUrl = `ws://localhost:8000/ws`;
    const websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setIsConnecting(false);
      setWs(websocket);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setVideoFeed(`data:image/jpeg;base64,${data.frame}`);
      if (data.prediction) {
        setPrediction(data.prediction);
      }
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setWs(null);
      setIsConnecting(false);
      // Attempt to reconnect after 2 seconds
      setTimeout(connectWebSocket, 2000);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      websocket.close();
    };
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [connectWebSocket]);

  const handleCapture = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send('capture');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-indigo-900 mb-2">
            Silent Voice - A Sign Language Detection
          </h1>
          <p className="text-gray-600">
            By Team Asteroid Destroyers!
          </p>
        </header>

        <div className="bg-white rounded-2xl shadow-xl p-6">
          <div className="relative">
            {isConnecting && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-90 rounded-lg">
                <div className="flex items-center space-x-3">
                  <RefreshCw className="w-6 h-6 text-indigo-600 animate-spin" />
                  <span className="text-indigo-600 font-medium">
                    Connecting to camera...
                  </span>
                </div>
              </div>
            )}
            <img
              src={videoFeed || 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3'}
              alt="Video Feed"
              className="w-full rounded-lg shadow-inner"
              style={{ minHeight: '400px', objectFit: 'cover' }}
            />
          </div>

          <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex-1 w-full sm:w-auto">
              <div className="bg-gray-50 rounded-lg p-4">
                <h2 className="text-sm font-medium text-gray-500 mb-1">
                  Current Prediction
                </h2>
                <p className="text-2xl font-bold text-indigo-600">
                  {prediction || 'No gesture detected'}
                </p>
              </div>
            </div>

            <button
              onClick={handleCapture}
              className="w-full sm:w-auto px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium flex items-center justify-center gap-2 hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!ws || ws.readyState !== WebSocket.OPEN}
            >
              <Camera className="w-5 h-5" />
              Capture Gesture
            </button>
          </div>
        </div>

        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Position your hand in the green box for best detection results</p>
        </footer>
      </div>
    </div>
  );
}

export default App;