import React, { useState } from "react";
import { getBestBuild } from "./services/api";
import BuildResult from "./components/BuildResult";


const App: React.FC = () => {
  const [items, setItems] = useState<any[]>([]);
  const [score, setScore] = useState<number | null>(null);
  const [stats, setStats] = useState<any | null>(null);

  const handleClick = async () => {
    try {
      const data = await getBestBuild();
      console.log("Données reçues du backend :", data);

      setItems(data.items || []);   // ⚡ récupère le tableau d'items
      setScore(data.score || null); // ⚡ récupère le score
      setStats(data.stats || null); // ⚡ récupère les stats
    } catch (error) {
      console.error("Erreur API:", error);
    }
  };

  return (
    <div className="p-6">
      <button
        onClick={handleClick}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg"
      >
        Obtenir le meilleur build
      </button>

      {score !== null && <p className="mt-4">Score du build : {score}</p>}

      {stats && (
        <div className="mt-4 p-4 bg-gray-200 rounded">
          <h2 className="font-bold">Stats globales :</h2>
          <pre>{JSON.stringify(stats, null, 2)}</pre>
        </div>
      )}

      <BuildResult items={items} />
    </div>
  );
};

export default App;