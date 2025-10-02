import React from "react";
import { rarityColors } from "./rarityColors";


interface Item {
  id: number;
  nom: string;
  type?: string;
  rarete?: string;
  niveau: number;
  pa?: number;
  pm?: number;
  pw?: number;
  portee?: number;
  controle?: number;
  pv?: number;
  coup_critique?: number;
  maitrise_melee?: number;
  maitrise_distance?: number;
  maitrise_berserk?: number;
  maitrise_critique?: number;
  maitrise_dos?: number;
  maitrise_1_element?: number;
  maitrise_2_elements?: number;
  maitrise_3_elements?: number;
  maitrise_elementaire?: number;
  maitrise_feu?: number;
  maitrise_eau?: number;
  maitrise_terre?: number;
  maitrise_air?: number;
  maitrise_soin?: number;
  tacle?: number;
  esquive?: number;
  initiative?: number;
  parade?: number;
  resistance_elementaire?: number;
  resistance_1_element?: number;
  resistance_2_elements?: number;
  resistance_3_elements?: number;
  resistance_feu?: number;
  resistance_eau?: number;
  resistance_terre?: number;
  resistance_air?: number;
  resistance_critique?: number;
  resistance_dos?: number;
  armure_donnee?: number;
  armure_recue?: number;
  volonte?: number;
  effets_supplementaires?: string;
  [key: string]: any; // pour capturer toutes les autres stats possibles
}

interface ItemCardProps {
  item: Item;
}

const ItemCard: React.FC<ItemCardProps> = ({ item }) => {
  // On applique la couleur selon la rareté
  const rarityClass =
  rarityColors[item.rarete?.toLowerCase() || ""] || "border-gray-200 bg-white";

  // On filtre uniquement les stats non nulles / non undefined
  const stats = Object.entries(item).filter(
  ([key, value]) => !["id", "nom", "type", "rarete", "niveau"].includes(key) && 
  value !== null && value !== 0 && value !== undefined
  );

  return (
    <div className={`p-4 rounded-2xl shadow-md border-2 transition-transform hover:scale-105 ${rarityClass}`}>
      <h3 className="text-lg font-bold text-gray-800">{item.nom}</h3>
      <p className="text-sm text-gray-500">
        Niveau {item.niveau} {item.type ? `• ${item.type}` : ""}{" "}
        {item.rarete ? `• ${item.rarete}` : ""}
      </p>

      <ul className="mt-3 space-y-1 text-sm">
        {stats.map(([key, value]) => (
          <li key={key} className="flex justify-between">
            <span className="capitalize text-gray-700">{key.replace(/_/g, " ")}:</span>
            <span className="font-medium text-gray-900">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ItemCard;