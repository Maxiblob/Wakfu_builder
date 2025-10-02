import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;

const api = axios.create({
  baseURL: API_URL, // proxy vers backend
});

export async function getBestBuild(level: number = 245) {
  const res = await api.get("/builds/optimise/" + level);
  return res.data;
}
