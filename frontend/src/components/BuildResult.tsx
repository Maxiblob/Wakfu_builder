import React from "react";
import ItemCard from "./ItemCard";

interface Item {
  id: number;
  nom: string;
  niveau: number;
  [key: string]: any;
}

interface BuildResultProps {
  items: Item[];
}

const BuildResult: React.FC<BuildResultProps> = ({ items }) => {
  if (!items || items.length === 0) {
    return <p>Aucun résultat disponible</p>;
  }

  return (
    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {items.map((item) => (
        <ItemCard key={item.id} item={item} />
      ))}
    </div>
  );
};

export default BuildResult;