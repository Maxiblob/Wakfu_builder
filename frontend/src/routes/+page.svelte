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

  const statOptions = [
    { key: "pa", label: "PA" },
    { key: "pm", label: "PM" },
    { key: "pw", label: "PW" },
    { key: "portee", label: "Portee" },
    { key: "controle", label: "Controle" },
    { key: "pv", label: "Points de vie" },
    { key: "coup_critique", label: "Coup critique" },
    { key: "maitrise_melee", label: "Maitrise melee" },
    { key: "maitrise_distance", label: "Maitrise distance" },
    { key: "maitrise_berserk", label: "Maitrise berserk" },
    { key: "maitrise_critique", label: "Maitrise critique" },
    { key: "maitrise_dos", label: "Maitrise dos" },
    { key: "maitrise_1_element", label: "Maitrise 1 element" },
    { key: "maitrise_2_elements", label: "Maitrise 2 elements" },
    { key: "maitrise_3_elements", label: "Maitrise 3 elements" },
    { key: "maitrise_elementaire", label: "Maitrise elementaire" },
    { key: "maitrise_feu", label: "Maitrise feu" },
    { key: "maitrise_eau", label: "Maitrise eau" },
    { key: "maitrise_terre", label: "Maitrise terre" },
    { key: "maitrise_air", label: "Maitrise air" },
    { key: "maitrise_soin", label: "Maitrise soin" },
    { key: "tacle", label: "Tacle" },
    { key: "esquive", label: "Esquive" },
    { key: "initiative", label: "Initiative" },
    { key: "parade", label: "Parade" },
    { key: "resistance_elementaire", label: "Resistance elementaire" },
    { key: "resistance_1_element", label: "Resistance 1 element" },
    { key: "resistance_2_elements", label: "Resistance 2 elements" },
    { key: "resistance_3_elements", label: "Resistance 3 elements" },
    { key: "resistance_feu", label: "Resistance feu" },
    { key: "resistance_eau", label: "Resistance eau" },
    { key: "resistance_terre", label: "Resistance terre" },
    { key: "resistance_air", label: "Resistance air" },
    { key: "resistance_critique", label: "Resistance critique" },
    { key: "resistance_dos", label: "Resistance dos" },
    { key: "armure_donnee", label: "Armure donnee" },
    { key: "armure_recue", label: "Armure recue" },
    { key: "volonte", label: "Volonte" },
  ];

  let level = 230;
  let selectedStats: string[] = [];
  let effectiveMasteryInput = "";
  let effectiveSelection: string[] = [];
  let effectiveWeight = 30;
  let zeroComponentWeights = false;
  let forceLegendary = false;
  let requireEpic = false;
  let requireRelic = false;
  let solver: "ga" | "cp" = "cp";
  let prioritizePA = false;
  let prioritizePM = false;
  let bannedIds: number[] = [];
  let bannedNames: string[] = [];
  let topK = 25;
  let popSize = 60;
  let generations = 80;
  let elite = 3;
  let probaMutation = 0.2;
  let loading = false;
  let error: string | null = null;
  let score: number | null = null;
  let stats: Record<string, number> | null = null;
  let items: Item[] = [];
  type Build = { score: number | null; stats: Record<string, number> | null; items: Item[] };
  let builds: Build[] = [];
  let selectedBuild = 0;
  let copyMessage = "";
  let copiedItemId: number | null = null;

  $: currentBuild = builds[selectedBuild] ?? { score, stats, items };
  $: currentScore = currentBuild?.score ?? null;
  $: currentStats = currentBuild?.stats ?? null;
  $: currentItems = currentBuild?.items ?? [];

  const readableKey = (key: string) => key.replace(/_/g, " ");

  const badgeColor = (rarete: string | undefined) => {
    if (!rarete) return "#e5e7eb";
    return rarityColors[rarete.toLowerCase()] ?? "#e5e7eb";
  };

  const toggleStat = (key: string) => {
    selectedStats = selectedStats.includes(key)
      ? selectedStats.filter((stat) => stat !== key)
      : [...selectedStats, key];
  };

  const toggleEffective = (key: string) => {
    effectiveSelection = effectiveSelection.includes(key)
      ? effectiveSelection.filter((s) => s !== key)
      : [...effectiveSelection, key];
  };

  async function generateBuild() {
    loading = true;
    error = null;
    try {
      const params = new URLSearchParams();
      selectedStats.forEach((stat) => params.append("stats", stat));
      // maîtrise effective (liste séparée par virgules ou points-virgules + sélection)
      const effectiveList = [
        ...effectiveSelection,
        ...effectiveMasteryInput
          .split(/[,;]+/)
          .map((s) => s.trim())
          .filter(Boolean),
      ];
      // dédoublonne la liste
      Array.from(new Set(effectiveList)).forEach((stat) =>
        params.append("effective_mastery", stat)
      );

      params.set("effective_weight", String(effectiveWeight));
      params.set("zero_component_weights", String(zeroComponentWeights));
      params.set("force_legendary", String(forceLegendary));
      params.set("require_epic", String(requireEpic));
      params.set("require_relic", String(requireRelic));
      params.set("prioritize_pa", String(prioritizePA));
      params.set("prioritize_pm", String(prioritizePM));
      bannedIds.forEach((id) => params.append("ban_ids", String(id)));
      bannedNames.forEach((name) => params.append("ban_names", name));
      params.set("solver", solver);
      if (solver === "cp") {
        params.set("verbose", "true");
      }
      params.set("top_k", String(topK));
      params.set("pop_size", String(popSize));
      params.set("generations", String(generations));
      params.set("elite", String(elite));
      params.set("proba_mutation", String(probaMutation));
      const query = params.toString();

      const res = await fetch(
        `${API_URL}/builds/optimise/${level}${query ? `?${query}` : ""}`
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      score = data.score ?? null;
      stats = data.stats ?? null;
      items = data.items ?? [];
      const alternatives = (data.alternatives as Build[] | undefined) ?? [];
      builds = [{ score, stats, items }, ...alternatives];
      selectedBuild = 0;
    } catch (e) {
      console.error(e);
      error = "Impossible de récupérer un build. Vérifie que l'API est en ligne.";
      score = null;
      stats = null;
      items = [];
    } finally {
      loading = false;
      copyMessage = "";
    }
  }

  async function copyItemName(name: string) {
    try {
      await navigator.clipboard.writeText(name);
      copyMessage = `Nom copié: ${name}`;
    } catch (e) {
      console.error("Clipboard error:", e);
      copyMessage = "Impossible de copier dans le presse-papier.";
    }
    setTimeout(() => {
      copyMessage = "";
      copiedItemId = null;
    }, 2000);
  }

  function banItem(item: Item) {
    const id = (item as any).id;
    const nom = String((item as any).nom ?? "");
    if (id !== undefined && id !== null && !bannedIds.includes(Number(id))) {
      bannedIds = [...bannedIds, Number(id)];
    } else if (nom && !bannedNames.includes(nom)) {
      bannedNames = [...bannedNames, nom];
    }
  }

  function clearBanned() {
    bannedIds = [];
    bannedNames = [];
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
    <div class="stat-selector">
      <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
        <p class="muted" style="margin: 0;">Stats a maximiser (optionnel)</p>
        {#if selectedStats.length}
          <span class="tag">{selectedStats.length} selectionnees</span>
        {/if}
      </div>
      <div class="chip-grid">
        {#each statOptions as option}
          <button
            class={`chip ${selectedStats.includes(option.key) ? "selected" : ""}`}
            type="button"
            on:click={() => toggleStat(option.key)}
          >
            <span>{option.label}</span>
          </button>
        {/each}
      </div>
    </div>

    <div class="card" style="padding: 1rem; margin-top: 1rem; border: 1px solid #e5e7eb; gap: 0.75rem; display: flex; flex-direction: column;">
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.75rem;">
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          Maîtrise effective (liste, séparée par virgules)
          <input
            class="input"
            placeholder="maitrise_distance, maitrise_3_elements, maitrise_feu..."
            bind:value={effectiveMasteryInput}
          />
        </label>
        <div class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          Maîtrises à inclure (sélection)
          <div class="chip-grid">
            {#each statOptions.filter((o) => o.key.startsWith("maitrise")) as option}
              <button
                class={`chip ${effectiveSelection.includes(option.key) ? "selected" : ""}`}
                type="button"
                on:click={() => toggleEffective(option.key)}
              >
                <span>{option.label}</span>
              </button>
            {/each}
          </div>
        </div>
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          Poids maîtrise effective
          <input class="input" type="number" step="0.5" bind:value={effectiveWeight} />
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          <input type="checkbox" bind:checked={zeroComponentWeights} />
          Ignorer le poids des composantes
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          <input type="checkbox" bind:checked={forceLegendary} />
          Forcer les items légendaires
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          <input type="checkbox" bind:checked={requireEpic} />
          Exiger un item épique
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          <input type="checkbox" bind:checked={requireRelic} />
          Exiger un item relique
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          <input type="checkbox" bind:checked={prioritizePA} />
          Prioriser les PA
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600;">
          <input type="checkbox" bind:checked={prioritizePM} />
          Prioriser les PM
        </label>
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          Mode d'optimisation
          <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
            <button
              type="button"
              class={`chip ${solver === "ga" ? "selected" : ""}`}
              on:click={() => (solver = "ga")}
            >
              Génétique
            </button>
            <button
              type="button"
              class={`chip ${solver === "cp" ? "selected" : ""}`}
              on:click={() => (solver = "cp")}
            >
              Contraintes (CP-SAT)
            </button>
          </div>
        </label>
      </div>

      <div style="margin-top: 0.25rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.5rem;">
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          top_k
          <input class="input" type="number" min="1" max="200" bind:value={topK} />
        </label>
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          population
          <input class="input" type="number" min="2" max="500" bind:value={popSize} />
        </label>
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          générations
          <input class="input" type="number" min="1" max="500" bind:value={generations} />
        </label>
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          elite
          <input class="input" type="number" min="1" max="20" bind:value={elite} />
        </label>
        <label class="muted" style="display: flex; flex-direction: column; gap: 0.25rem;">
          proba mutation
          <input class="input" type="number" min="0" max="1" step="0.05" bind:value={probaMutation} />
        </label>
      </div>

      <div class="card" style="padding: 0.75rem; border: 1px dashed #cbd5e1;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
          <span class="tag">Objets bannis</span>
          <button type="button" class="chip" on:click={clearBanned}>Réinitialiser</button>
        </div>
        {#if bannedIds.length === 0 && bannedNames.length === 0}
          <p class="muted" style="margin: 0;">Aucun objet banni. Clique sur "Bannir" dans une carte pour l'ajouter.</p>
        {:else}
          <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
            {#each bannedIds as id}
              <span class="chip selected">ID {id}</span>
            {/each}
            {#each bannedNames as name}
              <span class="chip selected">{name}</span>
            {/each}
          </div>
        {/if}
      </div>
    </div>
    {#if error}
      <p style="color: #dc2626; margin-top: 0.75rem;">{error}</p>
    {/if}
  </section>

  {#if builds.length > 1}
    <div class="card" style="padding: 0.75rem 1rem; border: 1px solid #e5e7eb;">
      <p class="muted" style="margin: 0 0 0.5rem 0;">Sélection du build</p>
      <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
        {#each builds as build, idx}
          <button
            type="button"
            class={`chip ${idx === selectedBuild ? "selected" : ""}`}
            on:click={() => (selectedBuild = idx)}
          >
            {idx === 0 ? "Build principal" : `Alternative ${idx}`}
            {#if build.score !== null}
              <span style="font-weight: 600;">(score {build.score})</span>
            {/if}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  {#if copyMessage}
    <div class="card" style="padding: 0.75rem 1rem; border: 1px solid #d1fae5; background: #ecfdf3; color: #065f46;">
      {copyMessage}
    </div>
  {/if}

  {#if currentScore !== null}
    <section class="card" style="padding: 1rem 1.25rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span class="tag">Score</span>
        <span style="font-weight: 700; font-size: 1.25rem;">{currentScore}</span>
      </div>
    </section>
  {/if}

  {#if currentStats}
    <section class="card" style="padding: 1rem 1.25rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span class="tag">Stats globales</span>
      </div>
      <div class="stats-panel">
        <pre>{JSON.stringify(currentStats, null, 2)}</pre>
      </div>
    </section>
  {/if}

  <section class="space-y-3">
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <h2 style="margin: 0;">Equipements</h2>
      {#if currentItems.length}
        <span class="muted">({currentItems.length} items)</span>
      {/if}
    </div>
    {#if currentItems.length === 0}
      <p class="muted">Aucun resultat. Lance une generation pour voir un build.</p>
    {:else}
      <div class="grid">
        {#each currentItems as item}
          {#if item}
            {@const rare = (item as any).rarete as string | undefined}
            <button
              type="button"
              class="item-card"
              on:click={() => {
                copiedItemId = (item as any).id ?? null;
                copyItemName((item as any).nom);
              }}
              style="cursor: pointer; text-align: left;"
              aria-label={`Copier ${String((item as any).nom ?? "l'item")}`}
            >
              {#if copiedItemId === (item as any).id && copyMessage}
                <div style="margin-bottom: 0.35rem; color: #065f46; background: #ecfdf3; border: 1px solid #d1fae5; padding: 0.4rem 0.6rem; border-radius: 8px; font-size: 0.9rem;">
                  {copyMessage}
                </div>
              {/if}
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
              <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <span
                  role="button"
                  tabindex="0"
                  class="chip"
                  on:click|stopPropagation={() => banItem(item)}
                  on:keydown|stopPropagation={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      banItem(item);
                    }
                  }}
                >
                  Bannir
                </span>
              </div>
            </button>
          {/if}
        {/each}
      </div>
    {/if}
  </section>
</main>
