<script lang="ts">
  type Item = Record<string, unknown>;

  const rarityColors: Record<string, string> = {
    inhabituel: "#9ca3af",
    rare: "#22c55e",
    mythique: "#f97316",
    legendaire: "#facc15",
    légendaire: "#facc15",
    epique: "#ec4899",
    épique: "#ec4899",
    relique: "#8b5cf6",
    souvenir: "#0ea5e9",
  };

  const API_URL =
    (import.meta as any).env?.PUBLIC_API_URL ??
    (import.meta as any).env?.VITE_API_URL ??
    "http://localhost:8000";

  let level = 200;
  let loading = false;
  let error: string | null = null;
  let score: number | null = null;
  let stats: Record<string, number> | null = null;
  let items: Item[] = [];

  const readableKey = (key: string) => key.replace(/_/g, " ");

  const badgeColor = (rarete: string | undefined) => {
    if (!rarete) return "#e5e7eb";
    return rarityColors[rarete.toLowerCase()] ?? "#e5e7eb";
  };

  async function generateBuild() {
    loading = true;
    error = null;
    try {
      const res = await fetch(`${API_URL}/builds/optimise/${level}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      score = data.score ?? null;
      stats = data.stats ?? null;
      items = data.items ?? [];
    } catch (e) {
      console.error(e);
      error = "Impossible de récupérer un build. Vérifie que l'API est en ligne.";
      score = null;
      stats = null;
      items = [];
    } finally {
      loading = false;
    }
  }
</script>

<main class="container space-y-6">
  <section class="card" style="padding: 1.5rem;">
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
      <div>
        <p class="muted" style="margin: 0;">Wakfu Build Optimizer</p>
        <h1 style="margin: 0; font-size: 1.5rem;">Genere un build par niveau</h1>
      </div>
      <div style="flex: 1;"></div>
      <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          Niveau
          <input
            class="input"
            type="number"
            min="1"
            max="245"
            bind:value={level}
          />
        </label>
        <button class="btn" on:click|preventDefault={generateBuild} disabled={loading}>
          {#if loading}
            Calcul en cours...
          {:else}
            Generer un build
          {/if}
        </button>
      </div>
    </div>
    {#if error}
      <p style="color: #dc2626; margin-top: 0.75rem;">{error}</p>
    {/if}
  </section>

  {#if score !== null}
    <section class="card" style="padding: 1rem 1.25rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span class="tag">Score</span>
        <span style="font-weight: 700; font-size: 1.25rem;">{score}</span>
      </div>
    </section>
  {/if}

  {#if stats}
    <section class="card" style="padding: 1rem 1.25rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span class="tag">Stats globales</span>
      </div>
      <div class="stats-panel">
        <pre>{JSON.stringify(stats, null, 2)}</pre>
      </div>
    </section>
  {/if}

  <section class="space-y-3">
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <h2 style="margin: 0;">Equipements</h2>
      {#if items.length}
        <span class="muted">({items.length} items)</span>
      {/if}
    </div>
    {#if items.length === 0}
      <p class="muted">Aucun resultat. Lance une generation pour voir un build.</p>
    {:else}
      <div class="grid">
        {#each items as item}
          {#if item}
            {@const rare = (item as any).rarete as string | undefined}
            <article class="item-card">
              <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                  <h3 style="margin: 0; font-size: 1.05rem;">{(item as any).nom}</h3>
                  <p class="muted" style="margin: 0.15rem 0 0;">
                    Niveau {(item as any).niveau}
                    {#if (item as any).type}
                      • {(item as any).type}
                    {/if}
                  </p>
                </div>
                {#if rare}
                  <span
                    class="tag"
                    style={`background:${badgeColor(rare)}22;color:${badgeColor(rare)};`}
                    >{rare}</span
                  >
                {/if}
              </div>

              <ul style="list-style: none; padding: 0; margin: 0.5rem 0 0; display: grid; gap: 0.35rem;">
                {#each Object.entries(item).filter(
                  ([key, value]) =>
                    !["nom", "type", "rarete", "niveau", "id"].includes(key) &&
                    value !== null &&
                    value !== undefined &&
                    value !== 0
                ) as [key, value]}
                  <li style="display: flex; justify-content: space-between; gap: 0.5rem;">
                    <span class="muted">{readableKey(key)}</span>
                    <span style="font-weight: 600;">{value as any}</span>
                  </li>
                {/each}
              </ul>
            </article>
          {/if}
        {/each}
      </div>
    {/if}
  </section>
</main>
