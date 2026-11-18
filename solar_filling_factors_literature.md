# Spot & facula filling factors on the Sun — observational relations from the literature

Compiled 2026-11-16. Each relation is labelled with how it was checked:
- **[V]** = read in the paper's full text during this compilation (coefficients copied from it).
- **[A]** = taken from the abstract or a secondary citation only. Check the paper before using the numbers.
- **[K]** = well-known value from background knowledge, not re-checked here.

Units:
- *Disc fraction* = projected area / πR².
- *MSH* (µHem) = 10⁻⁶ of the visible hemisphere (2πR²), corrected for foreshortening.
- Rule of thumb: for features spread over the visible disc, ⟨μ⟩ ≈ 0.5, so disc fraction ≈ A[MSH] × 10⁻⁶.

---

## 1. Spot and facular disc coverage vs. activity (solar cycle)

### 1a. Shapiro et al. 2014, A&A 569, A38 — vs. Ca II S-index [V]
Data: SATIRE-S coverages from KP/512, KP/SPM and SOHO/MDI, 1974–2009 roughly. They are binned in 58-day bins sorted by S-index (Sac Peak K-index converted to Mt Wilson S).

```
A_S(S) = (0.105 ± 0.011) − (1.315 ± 0.130)·S + (4.102 ± 0.370)·S²      (spots, disc fraction)
A_F(S) = −(0.233 ± 0.002) + (1.400 ± 0.010)·S                           (faculae, disc fraction)
```
- Solar minimum 1996 (annual S = 0.169): A_S ≈ 0.003 %, A_F ≈ 0.36 %.
- Solar maximum 1991 (annual S = 0.188): A_S ≈ 0.28 %, A_F ≈ 3 %.
- Faculae grow linearly with S and spots grow quadratically, so the facula/spot ratio falls as activity rises.
- KP/512 spot areas were multiplied by 1.127 to account for pores (Wenzler et al. 2006). Umbra : penumbra area is fixed at 1 : 4.

### 1b. Chatzistergos et al. 2022, A&A 667, A167 — Ca II K plage vs. sunspot area and number [V]
Data: 38 Ca II K archives from 1892 onward, in the CEA20 plage composite. Sunspot areas are from the MEA20 composite (Mandal et al. 2020). Areas are **projected disc fractions**. Fits are to daily values, restricted to days with sunspot area > 5×10⁻⁴ of the disc.

General form: `p_a = a1 + a2 · s^a3`

**Daily values (Table 4)**

| sunspot series s | a1 | a2 | a3 | RSS/DOF |
|---|---|---|---|---|
| MEA20 spot area | −0.004 ± 0.008 | 0.342 ± 0.093 | 0.35 ± 0.08 | 0.066 |
| ISN v1 (÷12.08) | 0.002 ± 0.002 | 0.009 ± 0.001 | 0.66 ± 0.05 | 0.116 |
| ISN v2 (×0.6/12.08) | 0.002 ± 0.002 | 0.009 ± 0.001 | 0.69 ± 0.04 | 0.074 |
| CEA17 group number | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.86 ± 0.06 | 0.078 |
| MEA20, square-root fit | 0.006 ± 0.002 | 0.606 ± 0.028 | 0.5 (fixed) | 0.110 |
| MEA20, linear fit | 0.020 ± 0.001 | 5.476 ± 0.261 | 1 (fixed) | 0.604 |

**Annual medians (Table 5)**: the relation becomes close to linear.

| s | a1 | a2 | a3 |
|---|---|---|---|
| MEA20 | 0.007 ± 0.001 | 0.0176 ± 0.0013 | 0.78 ± 0.06 |
| ISN v2 | 0.005 ± 0.001 | 0.0048 ± 0.0008 | 0.94 ± 0.07 |
| CEA17 GSN | 0.005 ± 0.001 | 0.0029 ± 0.0007 | 1.12 ± 0.10 |

**Cycle-strength-dependent versions (Eqs. A.1, A.2)**. ⟨s_a⟩ and ⟨s_gn⟩ are cycle-averaged values (Table A.1 below).
```
p_a = 0.34 · s_a^(0.565 − 53.3·⟨s_a⟩) − 0.004
p_a = 0.005 · s_gn^(0.75 + 0.021·⟨s_gn⟩) + 0.004
```
The plage/spot ratio therefore **decreases in stronger cycles**. The relation also depends on the Ca II K bandwidth: narrower passbands give steeper relations.

Worked example (daily Eq. 1):
- s_a = 2×10⁻³ gives p_a ≈ 0.035, a ratio of about 17.
- s_a = 2×10⁻⁴ gives p_a ≈ 0.013, a ratio of about 65.

**Cycle-averaged values (Table A.1)**. ⟨s_a⟩ is the MEA20 disc fraction × 1000. GSN is CEA17.

| SC | ⟨s_a⟩×10³ | ⟨GSN⟩ | SC | ⟨s_a⟩×10³ | ⟨GSN⟩ |
|---|---|---|---|---|---|
| 11 | 0.28 | 4.34 | 18 | 1.60 | 6.06 |
| 12 | 0.69 | 3.29 | 19 | 1.98 | 6.85 |
| 13 | 0.88 | 3.84 | 20 | 1.11 | 5.16 |
| 14 | 0.70 | 3.12 | 21 | 1.73 | 6.24 |
| 15 | 0.89 | 4.09 | 22 | 1.66 | 6.04 |
| 16 | 0.95 | 3.95 | 23 | 1.32 | 5.01 |
| 17 | 1.24 | 5.38 | 24 | 0.88 | – |

### 1c. Published facula(plage)-to-spot area ratios (from Chatzistergos 2022, Table 1) [V]

| study | data | ratio A_fac/A_spot |
|---|---|---|
| Chapman et al. 2001 | SFO CFDT1/2, 1988–1999 | ~5–40 |
| Schatten et al. 1985 | 1969–1982 | ~18–40 |
| Lawrence 1987 | 1974–1985 | ~5–30 |
| Steinegger et al. 1996 | VTT 1980 | ~3–30 |
| Borgniet et al. 2015 | MDI vs SOON | ~10–160 |
| Chatzistergos et al. 2020 | SP Ca II K vs Balmaceda areas | ~35–55 |
| Chatzistergos et al. 2020 | Kyoto Ca II K | ~13–20 |
| Foukal 1993/1996/1998 | MW/SGD vs RGO | linear (annual), quadratic/saturating (Foukal 1998) |
| Chapman et al. 1997; Solanki & Unruh 2013; Shapiro 2014 | — | quadratic |

Two robust conclusions:
1. On daily or rotational timescales, plage area ∝ (spot area)^~0.35–0.5, so the ratio falls with activity.
2. On annual means, the relation is close to linear.

### 1d. Extrapolation to more active stars
- **Nèmec et al. 2022, ApJL 934, L23** ("Faculae cancel out on the surfaces of active Suns") [A]. They use a surface flux transport model with the solar fit form `A_f = a·√(A_s + c) + d`; the coefficients are only in their Fig. 1 legend. The facular area grows sub-linearly: for emergence rates 16× and 64× solar, A_f is only about 14× and 50× solar.
- **Shapiro et al. 2016, A&A 589, A46** [K] and **Shapiro 2014**: the transition from facula-dominated to spot-dominated variability happens just above solar maximum activity (S ≈ 0.19–0.2).

### 1e. Spot area vs. sunspot number [K]
- Hathaway 2015 (Living Rev. Sol. Phys. 12, 4): total spot area ≈ 16.7 × R (ISN v1) in MSH. With ISN v2, divide by ~0.6.
- Daily relation (Chatzistergos 2022 [V]): from the tables above, the area–number relation is non-linear. This agrees with Balmaceda et al. 2009, Mandal et al. 2021 and Nagovitsyn & Osipova 2021.
- Number of groups (Jiang et al. 2011 [V]): N_groups/month ≈ R_G / 2.1. Işık et al. 2018 use R_G / 2.75 for bipolar magnetic regions.

---

## 2. Latitude distributions

### 2a. Sunspots — Jiang et al. 2011, A&A 528, A82 (RGO 1874–1976) [V]
x = fractional cycle phase (bin i / 30).
```
mean latitude:   λ(x)   = 26.4 − 34.2·x + 16.1·x²        [deg]   (rms 0.3°)
cycle strength:  λ_n    = 12.2 + 0.022·S_n                [deg]   r = 0.94
                 S_n = max of 12-month running mean of group sunspot number R_G
combined:        λ_n(x) = (26.4 − 34.2x + 16.1x²) · λ_n / 14.6°
latitude width:  σ(x)   = (0.14 + 1.05x − 0.78x²) · λ(x)  (Gaussian, truncated at 2σ)
tilt (Joy):      α = T_n·√|λ|,  T_n = 1.73 − 0.0035·S_n
group area PDF:  n(A) = 0.3·A^−1.1                 (A < 300 MSH)
                 n(A) = 0.003·exp[−(ln A − ln 45)²/(2 ln 3)]   (A > 60 MSH)
mean area:       A(x) = 115 + 396x − 426x²   [MSH]
```
Stronger cycles have spots at higher mean latitudes (+0.022° per unit of S_n). They also have smaller tilts.

### 2b. Sunspots — Hathaway 2011, Sol. Phys. 273, 221 ("standard law") [V]
```
λ_c(t) = 28° · exp[ −(t − t0) / 90 months ]
```
- Centroid latitude of sunspot area in each hemisphere.
- t0 is the fitted cycle start. It lies −8 to +3 months from minimum; small cycles start about 7 months before minimum.
- The path does **not** depend significantly on cycle amplitude or hemisphere. The drift slows and stops at about 7°.
- The latitude width (σ about the mean) is narrow at minimum, largest near maximum, and larger in big cycles. See also Hathaway 2015 (Living Review).

Note that Jiang 2011 (latitude scales with cycle strength) and Hathaway 2011 (no amplitude dependence once t0 is shifted) differ. Solanki, Wenzler & Schmitt 2008 (A&A 483, 623) [K] also find that early-cycle latitude correlates with cycle strength.

### 2c. Faculae / plage latitudes
- **Jha, Chatzistergos et al. 2024/25** (Kodaikanal Ca II K 1904–2007, arXiv:2408.13192; under review in Sol. Phys. at the time of the arXiv posting) [A]: the plage butterfly diagram follows Spörer's law. The mean plage latitude is **20.5 % ± 2.0 higher than that of sunspots**, whatever the cycle phase or strength. The north dominated in SC 19–20 and the south in SC 21–23.
- **Elek et al.** (SDO/SOHO faculae) [A]: faculae follow a butterfly diagram with quasi-biennial fine structure. For low-latitude faculae, the facula/spot area ratio is anti-correlated with spot area (cf. Chapman et al. 1997).
- **Polar faculae** (> 60–70°) are an anti-phase population, most numerous near cycle minimum.
  - Sheeley 1964, 1991, 2008 [K]: the number density of polar faculae tracks polar field strength, and the total count tracks polar flux.
  - Muñoz-Jaramillo et al. 2012, ApJ 753, 146 [A]: MWO polar facula counts for 1906–2014 are calibrated against WSO/MDI polar flux poleward of 70°.
- In practice, the facular latitude distribution is the spot distribution widened and shifted poleward by about 20 %, plus network and decayed-region flux that drifts poleward. A surface flux transport model reproduces this.

---

## 3. Spot size distribution (how many spots, how big)

**Baumann & Solanki 2005, A&A 443, 1061** (RGO 1874–1976, |CMD| < 30°) [V]. Log-normal fit:
```
dN/dA ∝ exp[ −(ln A − ln⟨A⟩)² / (2 ln σ_A) ]
```

| sample | method | ⟨A⟩ [MSH] | σ_A |
|---|---|---|---|
| umbrae | maximum area | 11.8 | 2.55 |
| umbrae | snapshot | 12.0 | 2.24 |
| total (umbra + penumbra) | maximum area | 62.2 | 2.45 |
| total | snapshot | 58.6 | 2.49 |
| single spots, total | maximum area | 45.5 | 2.11 |
| single spots, total | snapshot | 30.2 | 2.14 |
| Mt Wilson umbrae (Bogdan et al. 1988) | — | 0.62 | 3.80 |

Other size distributions:
- Hathaway & Choudhary 2008 (RGO + USAF/NOAA 1874–2007, A > 35 MSH) [A]: log-normal with similar parameters, and the shape is nearly independent of cycle phase.
- Harvey & Zwaan 1993; Schrijver & Harvey 1994 [K]: bipolar region flux/area spectrum ≈ power law n(A) ∝ A^−2 down to ephemeral regions. The amplitude scales with activity and the shape stays roughly fixed.
- Spot decay (Martínez Pillet et al. 1993, used by Baumann & Solanki): decay rate D is log-normal with μ = 1.75 and σ² = 2 (MSH/day).

---

## 4. Nesting (activity complexes) and active longitudes

### 4a. Observed nesting fractions

| study | data | result |
|---|---|---|
| Castenmiller, Zwaan & van der Zalm 1986, Sol. Phys. 105, 237 [A] | 1959–1964 | **30–40 %** of active regions are in nests. Nest = ≥ 3 regions within 4 rotations, ±7.5° longitude, ±5° latitude. Lifetime 6–15 rotations; nests do not spread. |
| Brouwer & Zwaan 1990, Sol. Phys. 129, 221 [A] | Greenwich | **≥ 1/3** of spot appearances belong to physical clusters. Spread ~2° in longitude × 1° in latitude; lifetime 1–7 months. |
| Schrijver & Zwaan 2000 (book) [A] | — | up to ~50 % of active regions emerge in nests |
| Pojoga & Cudnik 2002, Sol. Phys. [A] | SC 23 | used for nest member scatter: σ ≈ 2° latitude, 3° longitude (as adopted by Işık 2018) |
| **Karapınar, Işık, Krivova & Şenavcı 2026** (arXiv:2512.17364, DBSCAN clustering; RGO 1874–1976 + KMAS 1955–2025) [V] | SC 12–25 | **mean nesting degree D = N_nest/N = 0.61 ± 0.12**. See the per-cycle table below. |

Karapınar et al. 2026, per cycle (mean ± σ over 5° latitude windows):

| SC | D | SC | D |
|---|---|---|---|
| 12 | 0.552 ± 0.128 | 19 | 0.643 (RGO) / 0.684 (KMAS) |
| 13 | 0.539 ± 0.151 | 20 | 0.640 (RGO) / 0.661 (KMAS) |
| 14 | 0.560 ± 0.095 | 21 | 0.671 ± 0.103 |
| 15 | 0.593 ± 0.178 | 22 | 0.642 ± 0.072 |
| 16 | 0.610 ± 0.081 | 23 | 0.652 ± 0.091 |
| 17 | 0.587 ± 0.158 | 24 | 0.664 ± 0.063 |
| 18 | 0.590 ± 0.181 | 25 | 0.627 ± 0.101 |

Karapınar et al. 2026, other results:
- **By latitude:** D = 0.58 at 0–5°, peaks at **0.69 at 10–20°**, and drops to 0.42 at 30–35°.
- **By cycle strength:** D correlates with total group area per window, r = 0.60 (RGO) and 0.73 (KMAS). The correlation rises to r ≈ 0.78 for groups > 200 MSH.
- **Inter-nest spacing:** ~200–500 Mm at low activity, ~60–100 Mm at maximum, mean ~120 Mm. Nest lifetimes are one to several rotations.

### 4b. How nesting is modelled — Işık et al. 2018, A&A 620, A177 [V]
- Each new region has probability p of joining the current nest (a chain of Bernoulli draws). They used p = 0.7, described as deliberately high.
- Nest members are scattered with σ = 2° in latitude and 3° in longitude (after Pojoga & Cudnik 2002).
- Latitudes follow Jiang 2011 (§2a), with S☉ = 156. Each bipolar region polarity has a radius of 4°.
- **Nèmec et al. 2023, A&A** (Paper II) [A]: to reproduce the variability of most Kepler stars, nesting must be **much stronger than on the Sun**.

### 4c. Active longitudes [K]
- Berdyugina & Usoskin 2003 (A&A 405, 1121) and Usoskin et al. 2005: two persistent active longitudes ~180° apart, with "flip-flops" every ~3.7 yr.
- Jiang et al. 2011 [V] measured the non-random longitude part with an equatorial-dipole proxy: a mixing fraction c ≈ 0.15 of "ordered" longitudes over τ = 6 months.

---

## 5. Quick-look numbers (solar, disc fractions)

| quantity | minimum | maximum | source |
|---|---|---|---|
| spot coverage A_S | ~0–0.003 % | ~0.2–0.3 % (annual) | Shapiro 2014 [V] |
| facular coverage A_F (magnetogram) | ~0.36 % | ~3 % | Shapiro 2014 [V] |
| Ca II K plage area | ~0.4–0.7 % | ~3–5 % (daily, strong cycles) | Chatzistergos 2022 fits [V] |
| facula/spot ratio | ≫ 50 | ~10–20 | Chatzistergos 2022; Chapman 2001 |
| umbra : penumbra area | 1 : 4 (≈ 0.2 umbral fraction) | — | Wenzler 2006 via Shapiro 2014 [V] |
| mean spot latitude | ~26–28° at cycle start | ~8° at cycle end | Jiang 2011; Hathaway 2011 [V] |
| plage latitude | ~1.2 × spot latitude | — | Jha et al. 2024 [A] |
| nested fraction | 0.3–0.4 (classic) → 0.61 ± 0.12 (2026 clustering) | higher in strong cycles | §4a |

---

## References (links)
- Chatzistergos et al. 2022 — https://arxiv.org/abs/2209.07077 (A&A 667, A167)
- Shapiro et al. 2014 — https://arxiv.org/abs/1406.2383 (A&A 569, A38)
- Jiang, Cameron, Schmitt & Schüssler 2011 — https://arxiv.org/abs/1102.1266 (A&A 528, A82)
- Hathaway 2011 — https://arxiv.org/abs/1108.1722 (Sol. Phys. 273, 221)
- Hathaway 2015, The Solar Cycle — https://arxiv.org/abs/1502.07020
- Baumann & Solanki 2005 — https://arxiv.org/abs/astro-ph/0510516 (A&A 443, 1061)
- Işık et al. 2018 — https://arxiv.org/abs/1810.06728 (A&A 620, A177)
- Nèmec et al. 2022 — https://arxiv.org/abs/2207.06816 (ApJL 934, L23)
- Nèmec et al. 2023 — https://arxiv.org/abs/2303.03040 (A&A, 2023)
- Karapınar et al. 2026, nesting — https://arxiv.org/abs/2512.17364
- Jha et al., Kodaikanal plage butterfly — https://arxiv.org/abs/2408.13192
- Mandal et al. 2020, sunspot area composite — https://arxiv.org/abs/2004.14618 (A&A 640, A78)
- Muñoz-Jaramillo et al. 2012/2013, polar faculae — https://arxiv.org/abs/1303.0345 , https://arxiv.org/abs/1304.3151
- Latitudinal migration of sunspot groups (2024) — https://arxiv.org/abs/2405.15355
- Mean butterfly diagram & poloidal source (2024) — https://arxiv.org/abs/2405.17185
