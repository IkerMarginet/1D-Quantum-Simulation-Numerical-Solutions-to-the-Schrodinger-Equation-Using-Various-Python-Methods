"""
SIMULATION QUANTIQUE 1D
"""
 
# ======================================================================
# IMPORTATION DES PACKAGES
# ======================================================================
import numpy as np  # Pour les calculs numériques et tableaux
import matplotlib.pyplot as plt  # Pour la visualisation
from scipy.linalg import eigh_tridiagonal  # Pour diagonaliser les matrices tridiagonales
from matplotlib.animation import FuncAnimation  # Pour créer des animations

"""
Pourquoi ces packages ?
- numpy: essentiel pour le calcul scientifique en Python (opérations vectorisées)
- matplotlib: standard pour la visualisation scientifique
- scipy.linalg: fournit des algorithmes optimisés pour l'algèbre linéaire
- FuncAnimation: permet de créer des animations fluides des résultats
"""

# ======================================================================
# CONSTANTES PHYSIQUES (unités naturelles)
# ======================================================================
hbar = 1.0  # Constante de Planck réduite (unités naturelles)
m = 1.0     # Masse de la particule
omega = 1.0 # Fréquence angulaire pour l'oscillateur harmonique

"""
Unités naturelles:
Nous utilisons un système où hbar = m = 1 pour simplifier les calculs.
Cela signifie que toutes les énergies sont en multiples de hbar*omega et 
les longueurs en multiples de sqrt(hbar/(m*omega)).
"""

# ======================================================================
# DÉFINITION DES POTENTIELS QUANTIQUES
# ======================================================================
def potentiel_harmonique(x, omega=1.0):
    """
    Potentiel de l'oscillateur harmonique quantique
    V(x) = (1/2) * m * omega^2 * x^2
    
    Derivation:
    C'est le potentiel quadratique standard qui apparaît dans l'approximation 
    harmonique autour d'un minimum d'énergie potentielle.
    """
    return 0.5 * m * omega**2 * x**2

def potentiel_puits(x, V0=50, largeur=2):
    """
    Puits de potentiel fini rectangulaire
    V(x) = 0 si |x| < largeur/2, V0 sinon
    
    Physique:
    Modèle standard pour les états liés dans un puits fini.
    V0 représente la hauteur des barrières.
    """
    return np.where(np.abs(x) > largeur/2, V0, 0)

def potentiel_barriere(x, V0=50, largeur=2):
    """
    Barrière de potentiel rectangulaire
    V(x) = V0 si |x| < largeur/2, 0 sinon
    
    Application:
    Étude de l'effet tunnel et des coefficients de transmission.
    """
    return np.where(np.abs(x) < largeur/2, V0, 0)

def potentiel_double_puits(x, V0=50, largeur_puits=1, distance=3):
    """
    Double puits séparés par une barrière
    Deux puits de largeur 'largeur_puits' séparés par 'distance'
    
    Importance:
    Système modèle pour les transitions quantiques et l'effet tunnel.
    """
    dans_puits = (np.abs(x - distance/2) < largeur_puits/2) | (np.abs(x + distance/2) < largeur_puits/2)
    return np.where(dans_puits, 0, V0)

def potentiel_hydrogene_radial(r, l=0):
    """
    Potentiel coulombien radial avec terme centrifuge
    V(r) = -1/r + l(l+1)/(2r^2)
    
    Théorie:
    Ce potentiel modélise l'atome d'hydrogène en coordonnées radiales.
    Le terme centrifuge apparaît pour l > 0 (orbitales p, d, etc.).
    En unités atomiques (hbar = m = e = 1), le potentiel est simplifié.
    """
    return -1.0 / r + (l * (l + 1)) / (2 * r**2)

# ======================================================================
# CONSTRUCTION DU HAMILTONIEN
# ======================================================================
def construire_hamiltonien(N, x, potentiel):
    """
    Construit la matrice Hamiltonienne sous forme tridiagonale avec différences finies.
    
    Méthode numérique:
    On discrétise le Laplacien avec des différences finies centrées.
    Le terme cinétique donne une matrice tridiagonale avec 2 sur la diagonale
    et -1 sur les sous/sur-diagonales.
    
    Formules:
    d^2ψ/dx^2 ≈ (ψ_{n+1} - 2ψ_n + ψ_{n-1})/dx^2
    H = T + V = -hbar^2/(2m) d^2/dx^2 + V(x)
    """
    dx = x[1] - x[0]
    terme_cinetique = hbar**2 / (2 * m * dx**2)
    diagonale_principale = 2 * terme_cinetique + potentiel(x)
    diagonale_secondaire = -terme_cinetique * np.ones(N-1)
    return diagonale_principale, diagonale_secondaire

def construire_hamiltonien_numerov(N, x, potentiel):
    """
    Construit la matrice Hamiltonienne avec la méthode de Numerov.
    
    Méthode:
    Approximation d'ordre O(h^6) pour le Laplacien, adaptée à l'équation de Schrödinger.
    """
    dx = x[1] - x[0]
    V = potentiel(x)
    terme_cinetique = hbar**2 / (2 * m * dx**2)
    diag = (10/12) * V + 2 * terme_cinetique
    subdiag = (1/12) * V[:-1] - terme_cinetique
    diag[0] = 1e10  # Condition aux limites ψ=0
    subdiag[0] = 0.0
    diag[-1] = 1e10
    return diag, subdiag

def calculer_etats_propres(diag, subdiag):
    """
    Diagonalisation de l'Hamiltonien tridiagonal
    
    Algorithme:
    Utilise eigh_tridiagonal de SciPy qui est optimisé pour les matrices
    symétriques tridiagonales (O(N) au lieu de O(N^3) pour une matrice pleine).
    
    Retourne:
    - Valeurs propres (énergies) triées
    - Vecteurs propres (fonctions d'onde) normalisés
    """
    E, psi = eigh_tridiagonal(diag, subdiag)
    idx = np.argsort(E)
    return np.real(E[idx]), psi[:, idx]

def simpson_integration(y, x):
    """
    Implémentation personnalisée de l'intégration numérique par la méthode de Simpson.
    
    Paramètres:
    y: tableau des valeurs de la fonction à intégrer (par ex. |ψ(x)|^2)
    x: tableau des points de discrétisation (doit être uniformément espacé)
    
    Retourne:
    La valeur de l'intégrale calculée sur l'intervalle [x[0], x[-1]].
    
    Note:
    - Nécessite un nombre impair de points (nombre pair de sous-intervalles).
    - Si le nombre de points est pair, on utilise la méthode des trapèzes pour le dernier intervalle.
    """
    N = len(x)
    if N < 3:
        raise ValueError("La méthode de Simpson nécessite au moins 3 points.")
    
    dx = x[1] - x[0]
    
    if N % 2 == 0:
        result = 0.0
        for i in range(0, N-2, 2):
            result += y[i] + 4*y[i+1] + y[i+2]
        result *= dx / 3.0
        result += (dx / 2.0) * (y[N-2] + y[N-1])
    else:
        result = y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-2:2])
        result *= dx / 3.0
    
    return result

def normaliser_fonction_onde(psi, x):
    """
    Normalisation des fonctions d'onde
    
    Condition:
    ∫|ψ(x)|^2 dx = 1 pour chaque état
    
    Méthode:
    On utilise une implémentation personnalisée de la méthode de Simpson pour l'intégration numérique
    """
    for i in range(psi.shape[1]):
        densite = np.abs(psi[:, i])**2
        norme = np.sqrt(simpson_integration(densite, x))
        if norme == 0:
            raise ValueError(f"Norme nulle pour l'état {i}. Vérifiez la fonction d'onde.")
        psi[:, i] /= norme
    return psi

# ======================================================================
# VISUALISATION COMPLÈTE
# ======================================================================
def visualiser_systeme(x, V, E, psi, titre, E_theo=None):
    """
    Visualisation complète du système quantique
    
    Organisation:
    1. Graphe supérieur: potentiel + niveaux + densités de probabilité animées
    2. Graphe inférieur: parties réelle/imaginaire des 4 premiers états
    
    Théorie:
    La densité de probabilité |ψ(x,t)|^2 évolue comme |ψ(x,0)|^2 puisque
    ψ(x,t) = ψ(x,0) exp(-iEt/hbar) et donc |ψ(x,t)|^2 = |ψ(x,0)|^2
    """
    fig, (ax_pot, ax_wave) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax_pot.plot(x, V, 'k-', linewidth=2, label='Potentiel V(x)')
    ax_pot.set_xlabel('Position x')
    ax_pot.set_ylabel('Énergie')
    ax_pot.set_title(titre)
    ax_pot.grid(alpha=0.3)
    
    ax_pot.set_xlim(x.min(), x.max())
    V_min, V_max = np.min(V), np.max(V)
    E_max = E[9] if len(E) > 9 else E[-1]
    ax_pot.set_ylim(min(V_min, E[0]) - 0.1 * abs(V_min), 
                   max(V_max, E_max) + 0.2 * abs(E_max))
    
    ax_wave.set_xlabel('Position x')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_title('Fonctions d\'onde (4 premiers états)')
    ax_wave.grid(alpha=0.3)
    
    ax_wave.set_xlim(x.min(), x.max())
    psi_max = np.max(np.abs(psi[:,:4]))
    ax_wave.set_ylim(-1.2 * psi_max, 1.2 * psi_max)
    
    couleurs = plt.cm.plasma(np.linspace(0, 1, 10))
    
    for n in range(10):
        if n < len(E):
            ax_pot.axhline(E[n], color=couleurs[n], linestyle='--', alpha=0.5,
                          label=f'État {n} (E={E[n]:.2f})')
            if E_theo is not None and n < len(E_theo):
                ax_pot.axhline(E_theo[n], color=couleurs[n], linestyle=':', alpha=0.3,
                              label=f'Théorie {n} (E={E_theo[n]:.2f})')
    
    lines_dens = []
    lines_reel = []
    lines_imag = []
    
    for n in range(10):
        if n < len(E):
            line, = ax_pot.plot([], [], color=couleurs[n])
            lines_dens.append(line)
            if n < 4:
                l_reel, = ax_wave.plot([], [], color=couleurs[n], label=f'Réel {n}')
                l_imag, = ax_wave.plot([], [], color=couleurs[n], linestyle=':', label=f'Imag {n}')
                lines_reel.append(l_reel)
                lines_imag.append(l_imag)
    
    ax_pot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_wave.legend()
    
    def init():
        for line in lines_dens + lines_reel + lines_imag:
            line.set_data([], [])
        return lines_dens + lines_reel + lines_imag
    
    def update(t):
        for n in range(10):
            if n < len(E):
                psi_t = psi[:, n] * np.exp(-1j * E[n] * t / hbar)
                scale = 2.0 * (1 + 0.1 * np.sin(0.5 * t))
                lines_dens[n].set_data(x, scale * np.abs(psi_t)**2 + E[n])
                if n < 4:
                    lines_reel[n].set_data(x, np.real(psi_t))
                    lines_imag[n].set_data(x, np.imag(psi_t))
        return lines_dens + lines_reel + lines_imag
    
    anim = FuncAnimation(fig, update, frames=np.linspace(0, 4*np.pi, 100),
                        init_func=init, interval=50, blit=True)
    
    plt.tight_layout()
    return anim

# ======================================================================
# ÉTUDES QUANTIQUES COMPLÈTES
# ======================================================================
def etude_oscillateur_harmonique():
    """
    Étude de l'oscillateur harmonique quantique avec différences finies et Numerov.
    
    Théorie:
    - Énergies: E_n = hbar*omega*(n + 1/2)
    - Fonctions d'onde: polynômes d'Hermite × gaussienne
    
    Visualisation:
    - Ajoute un graphique comparant les énergies (théoriques, différences finies, Numerov)
    """
    N = 500
    L = 10
    x = np.linspace(-L/2, L/2, N)
    
    # Différences finies
    diag, subdiag = construire_hamiltonien(N, x, lambda x: potentiel_harmonique(x, omega=1.0))
    E_diff, psi_diff = calculer_etats_propres(diag, subdiag)
    psi_diff = normaliser_fonction_onde(psi_diff, x)
    
    # Numerov
    diag_n, subdiag_n = construire_hamiltonien_numerov(N, x, lambda x: potentiel_harmonique(x, omega=1.0))
    E_numerov, psi_numerov = calculer_etats_propres(diag_n, subdiag_n)
    psi_numerov = normaliser_fonction_onde(psi_numerov, x)
    
    # Énergies théoriques
    E_theo = [hbar * omega * (n + 0.5) for n in range(10)]
    
    # Affichage des énergies
    print("\nComparaison des énergies (Oscillateur Harmonique):")
    for n in range(10):
        print(f"État {n}:")
        print(f"  Diff. finies: {E_diff[n]:.4f} (théo: {E_theo[n]:.4f}, erreur: {abs(E_diff[n]-E_theo[n]):.2e})")
        print(f"  Numerov     : {E_numerov[n]:.4f} (théo: {E_theo[n]:.4f}, erreur: {abs(E_numerov[n]-E_theo[n]):.2e})")
    
    # Graphique de comparaison des énergies
    plt.figure(figsize=(10, 6))
    n_states = np.arange(10)
    plt.plot(n_states, E_theo, 'k*-', label='Théorique', markersize=10)
    plt.plot(n_states, E_diff[:10], 'ro--', label='Différences finies', alpha=0.7)
    plt.plot(n_states, E_numerov[:10], 'bs--', label='Numerov', alpha=0.7)
    plt.title("Comparaison des énergies (Oscillateur Harmonique)")
    plt.xlabel("État quantique n")
    plt.ylabel("Énergie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Visualisation (Numerov)
    anim = visualiser_systeme(x, potentiel_harmonique(x), E_numerov, psi_numerov,
                            "Oscillateur Harmonique Quantique (Numerov)", E_theo)
    plt.show()
    return anim

def etude_puits_fini():
    """
    Étude d'un puits de potentiel fini avec différences finies et Numerov.
    
    Physique:
    - Nombre fini d'états liés dépendant de V0
    - Pas de solution théorique simple, comparaison entre méthodes
    """
    N = 500
    L = 12
    x = np.linspace(-L/2, L/2, N)
    V0 = 50
    largeur = 3
    
    # Différences finies
    diag, subdiag = construire_hamiltonien(N, x, lambda x: potentiel_puits(x, V0, largeur))
    E_diff, psi_diff = calculer_etats_propres(diag, subdiag)
    psi_diff = normaliser_fonction_onde(psi_diff, x)
    
    # Numerov
    diag_n, subdiag_n = construire_hamiltonien_numerov(N, x, lambda x: potentiel_puits(x, V0, largeur))
    E_numerov, psi_numerov = calculer_etats_propres(diag_n, subdiag_n)
    psi_numerov = normaliser_fonction_onde(psi_numerov, x)
    
    # Affichage des énergies
    print("\nÉnergies (Puits Fini):")
    for n in range(10):
        print(f"État {n}: Diff. finies = {E_diff[n]:.4f}, Numerov = {E_numerov[n]:.4f}, "
              f"Différence = {abs(E_diff[n]-E_numerov[n]):.2e}")
    
    # Graphique de comparaison des énergies (pas de théorique)
    plt.figure(figsize=(10, 6))
    n_states = np.arange(10)
    plt.plot(n_states, E_diff[:10], 'ro--', label='Différences finies', alpha=0.7)
    plt.plot(n_states, E_numerov[:10], 'bs--', label='Numerov', alpha=0.7)
    plt.title("Comparaison des énergies (Puits Fini)")
    plt.xlabel("État quantique n")
    plt.ylabel("Énergie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    anim = visualiser_systeme(x, potentiel_puits(x, V0, largeur), E_numerov, psi_numerov,
                            f"Puits de Potentiel Fini\nV0={V0}, L={largeur} (Numerov)")
    plt.show()
    return anim

def etude_barriere():
    """
    Étude d'une barrière de potentiel avec différences finies et Numerov.
    
    Application:
    - Effet tunnel quantique
    """
    N = 500
    L = 15
    x = np.linspace(-L/2, L/2, N)
    V0 = 50
    largeur = 2
    
    # Différences finies
    diag, subdiag = construire_hamiltonien(N, x, lambda x: potentiel_barriere(x, V0, largeur))
    E_diff, psi_diff = calculer_etats_propres(diag, subdiag)
    psi_diff = normaliser_fonction_onde(psi_diff, x)
    
    # Numerov
    diag_n, subdiag_n = construire_hamiltonien_numerov(N, x, lambda x: potentiel_barriere(x, V0, largeur))
    E_numerov, psi_numerov = calculer_etats_propres(diag_n, subdiag_n)
    psi_numerov = normaliser_fonction_onde(psi_numerov, x)
    
    print("\nÉnergies (Barrière):")
    for n in range(10):
        print(f"État {n}: Diff. finies = {E_diff[n]:.4f}, Numerov = {E_numerov[n]:.4f}, "
              f"Différence = {abs(E_diff[n]-E_numerov[n]):.2e}")
    
    # Graphique de comparaison des énergies (pas de théorique)
    plt.figure(figsize=(10, 6))
    n_states = np.arange(10)
    plt.plot(n_states, E_diff[:10], 'ro--', label='Différences finies', alpha=0.7)
    plt.plot(n_states, E_numerov[:10], 'bs--', label='Numerov', alpha=0.7)
    plt.title("Comparaison des énergies (Barrière)")
    plt.xlabel("État quantique n")
    plt.ylabel("Énergie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    anim = visualiser_systeme(x, potentiel_barriere(x, V0, largeur), E_numerov, psi_numerov,
                            f"Barrière de Potentiel\nV0={V0}, L={largeur} (Numerov)")
    plt.show()
    return anim

def etude_double_puits():
    """
    Étude d'un double puits quantique avec différences finies et Numerov.
    
    Phénomènes:
    - États symétriques/antisymétriques
    - Effet tunnel
    """
    N = 500
    L = 15
    x = np.linspace(-L/2, L/2, N)
    V0 = 50
    largeur_puits = 1
    distance = 4
    
    # Différences finies
    diag, subdiag = construire_hamiltonien(N, x,
                     lambda x: potentiel_double_puits(x, V0, largeur_puits, distance))
    E_diff, psi_diff = calculer_etats_propres(diag, subdiag)
    psi_diff = normaliser_fonction_onde(psi_diff, x)
    
    # Numerov
    diag_n, subdiag_n = construire_hamiltonien_numerov(N, x,
                     lambda x: potentiel_double_puits(x, V0, largeur_puits, distance))
    E_numerov, psi_numerov = calculer_etats_propres(diag_n, subdiag_n)
    psi_numerov = normaliser_fonction_onde(psi_numerov, x)
    
    print("\nÉnergies (Double Puits):")
    for n in range(10):
        print(f"État {n}: Diff. finies = {E_diff[n]:.4f}, Numerov = {E_numerov[n]:.4f}, "
              f"Différence = {abs(E_diff[n]-E_numerov[n]):.2e}")
    
    # Graphique de comparaison des énergies (pas de théorique)
    plt.figure(figsize=(10, 6))
    n_states = np.arange(10)
    plt.plot(n_states, E_diff[:10], 'ro--', label='Différences finies', alpha=0.7)
    plt.plot(n_states, E_numerov[:10], 'bs--', label='Numerov', alpha=0.7)
    plt.title("Comparaison des énergies (Double Puits)")
    plt.xlabel("État quantique n")
    plt.ylabel("Énergie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    anim = visualiser_systeme(x, potentiel_double_puits(x, V0, largeur_puits, distance),
                            E_numerov, psi_numerov,
                            f"Double Puits Quantique\nV0={V0}, L={largeur_puits}, d={distance} (Numerov)")
    plt.show()
    return anim

def etude_hydrogene_radial():
    """
    Étude de l'atome d'hydrogène en coordonnées radiales (l=0) avec différences finies et Numerov.
    
    Théorie:
    - Fonction d'onde radiale: u(r) = r * R(r)
    - Énergies: E_n = -0.5 / n^2 (unités atomiques)
    
    Visualisation:
    - Ajoute un graphique comparant les énergies (théoriques, différences finies, Numerov)
    """
    r_min = 1e-6
    r_max = 50.0
    N = 5000
    l = 0
    r = np.linspace(r_min, r_max, N)
    
    # Différences finies
    def construire_hamiltonien_radial(r, l=0):
        terme_cinetique = hbar**2 / (2 * m * (r[1] - r[0])**2)
        diagonale_principale = 2 * terme_cinetique + potentiel_hydrogene_radial(r, l)
        diagonale_secondaire = -terme_cinetique * np.ones(N-1)
        diagonale_principale[0] = 1e10
        diagonale_secondaire[0] = 0.0
        diagonale_principale[-1] = 1e10
        return diagonale_principale, diagonale_secondaire
    
    diag, subdiag = construire_hamiltonien_radial(r, l)
    E_diff, psi_diff = eigh_tridiagonal(diag, subdiag, select='i', select_range=(0, 4))
    psi_diff = normaliser_fonction_onde(psi_diff, r)
    
    # Numerov
    def construire_hamiltonien_radial_numerov(r, l=0):
        dr = r[1] - r[0]
        terme_cinetique = hbar**2 / (2 * m * dr**2)
        V = potentiel_hydrogene_radial(r, l)
        diagonale_principale = (10/12) * V + 2 * terme_cinetique
        diagonale_secondaire = (1/12) * V[:-1] - terme_cinetique
        diagonale_principale[0] = 1e10
        diagonale_secondaire[0] = 0.0
        diagonale_principale[-1] = 1e10
        return diagonale_principale, diagonale_secondaire
    
    diag_n, subdiag_n = construire_hamiltonien_radial_numerov(r, l)
    E_numerov, psi_numerov = eigh_tridiagonal(diag_n, subdiag_n, select='i', select_range=(0, 4))
    psi_numerov = normaliser_fonction_onde(psi_numerov, r)
    
    # Énergies théoriques
    E_theo = [-0.5 / (n + 1)**2 for n in range(5)]
    
    print("\nÉnergies (Hydrogène Radial, l=0):")
    for i, (E_d, E_n, E_t) in enumerate(zip(E_diff, E_numerov, E_theo)):
        print(f"État {i+1}:")
        print(f"  Diff. finies: E = {E_d:.6f} a.u. (théo: {E_t:.6f}, erreur: {abs(E_d - E_t):.2e})")
        print(f"  Numerov     : E = {E_n:.6f} a.u. (théo: {E_t:.6f}, erreur: {abs(E_n - E_t):.2e})")
    
    # Graphique de comparaison des énergies
    plt.figure(figsize=(10, 6))
    n_states = np.arange(5)
    plt.plot(n_states, E_theo, 'k*-', label='Théorique', markersize=10)
    plt.plot(n_states, E_diff, 'ro--', label='Différences finies', alpha=0.7)
    plt.plot(n_states, E_numerov, 'bs--', label='Numerov', alpha=0.7)
    plt.title("Comparaison des énergies (Hydrogène Radial, l=0)")
    plt.xlabel("État quantique n")
    plt.ylabel("Énergie (a.u.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for i in range(min(3, psi_numerov.shape[1])):
        plt.plot(r, psi_numerov[:, i], label=f'État {i+1}, E = {E_numerov[i]:.6f} a.u.')
    plt.title("Fonctions d'onde radiales u(r) = rR(r) pour l=0 (Numerov)")
    plt.xlabel("r (a.u.)")
    plt.ylabel("u(r)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for i in range(min(3, psi_numerov.shape[1])):
        R_r = psi_numerov[:, i] / r
        plt.plot(r, R_r, label=f'État {i+1}, E = {E_numerov[i]:.6f} a.u.')
    plt.title("Fonctions radiales R(r) pour l=0 (Numerov)")
    plt.xlabel("r (a.u.)")
    plt.ylabel("R(r)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ======================================================================
# FONCTIONS SPÉCIALES
# ======================================================================
def analyse_erreur_oscillateur():
    """
    Analyse de convergence numérique pour l'oscillateur harmonique
    
    Méthode:
    Compare les énergies numériques avec les solutions exactes pour
    différents nombres de points de discrétisation N.
    
    Visualisation:
    Utilise la même couleur pour chaque état, avec 'o' pour différences finies
    et 's' pour Numerov.
    """
    L = 10
    Ns = [200, 300, 400, 500, 1000, 1500]
    erreurs_diff = []
    erreurs_numerov = []

    print("\nAnalyse de convergence pour l'oscillateur harmonique:")
    
    for N in Ns:
        x = np.linspace(-L/2, L/2, N)
        # Différences finies
        diag, subdiag = construire_hamiltonien(N, x, lambda x: potentiel_harmonique(x, omega))
        E_num_diff, _ = calculer_etats_propres(diag, subdiag)
        # Numerov
        diag_n, subdiag_n = construire_hamiltonien_numerov(N, x, lambda x: potentiel_harmonique(x, omega))
        E_num_numerov, _ = calculer_etats_propres(diag_n, subdiag_n)
        E_theo = np.array([hbar * omega * (n + 0.5) for n in range(10)])
        
        erreur_diff = np.abs((E_num_diff[:10] - E_theo) / E_theo)
        erreur_numerov = np.abs((E_num_numerov[:10] - E_theo) / E_theo)
        erreurs_diff.append(erreur_diff)
        erreurs_numerov.append(erreur_numerov)
        
        print(f"N = {N:>4} | Erreur moyenne (Diff. finies): {np.mean(erreur_diff):.3e} | "
              f"Erreur moyenne (Numerov): {np.mean(erreur_numerov):.3e}")

    erreurs_diff = np.array(erreurs_diff)
    erreurs_numerov = np.array(erreurs_numerov)
    plt.figure(figsize=(10, 6))
    
    couleurs = plt.cm.plasma(np.linspace(0, 1, 10))
    for k in range(10):
        plt.plot(Ns, erreurs_diff[:, k], '-o', color=couleurs[k], label=f'État {k} (Diff. finies)', alpha=0.7)
        plt.plot(Ns, erreurs_numerov[:, k], '--s', color=couleurs[k], label=f'État {k} (Numerov)', alpha=0.7)
    
    plt.title("Convergence des énergies numériques\n(Oscillateur Harmonique)")
    plt.xlabel("Nombre de points N")
    plt.ylabel("Erreur relative")
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def analyse_erreur_tous_potentiels():
    """
    Analyse de convergence numérique pour tous les potentiels quantiques
    
    Méthode:
    - Osc. harmonique et hydrogène radial: comparaison aux solutions exactes
    - Autres potentiels: comparaison à une solution de référence (N=2000, différences finies)
    - Inclut différences finies et Numerov
    
    Visualisation:
    Utilise la même couleur pour chaque état, avec 'o' pour différences finies
    et 's' pour Numerov.
    """
    Ns = [200, 300, 400, 500, 1000, 1500]
    potentiels = [
        ("Oscillateur Harmonique", lambda x: potentiel_harmonique(x, omega=1.0), 10, True),
        ("Puits Fini", lambda x: potentiel_puits(x, V0=50, largeur=3), 12, False),
        ("Barrière", lambda x: potentiel_barriere(x, V0=50, largeur=2), 15, False),
        ("Double Puits", lambda x: potentiel_double_puits(x, V0=50, largeur_puits=1, distance=4), 15, False),
        ("Hydrogène Radial (l=0)", lambda r: potentiel_hydrogene_radial(r, l=0), 50, True)
    ]

    for nom, potentiel, L, theorie in potentiels:
        print(f"\nAnalyse de convergence pour {nom}:")
        erreurs_diff_finies = []
        erreurs_numerov = []
        
        if theorie:
            if nom == "Oscillateur Harmonique":
                E_ref = np.array([hbar * omega * (n + 0.5) for n in range(10)])
            else:
                E_ref = np.array([-0.5 / (n + 1)**2 for n in range(10)])
        else:
            x_ref = np.linspace(1e-6 if nom == "Hydrogène Radial (l=0)" else -L/2, 
                               L if nom == "Hydrogène Radial (l=0)" else L/2, 2000)
            diag_ref, subdiag_ref = construire_hamiltonien(2000, x_ref, potentiel)
            E_ref, _ = calculer_etats_propres(diag_ref, subdiag_ref)
            E_ref = E_ref[:10]

        for N in Ns:
            x = np.linspace(1e-6 if nom == "Hydrogène Radial (l=0)" else -L/2, 
                           L if nom == "Hydrogène Radial (l=0)" else L/2, N)
            
            diag, subdiag = construire_hamiltonien(N, x, potentiel)
            E_num_diff, _ = calculer_etats_propres(diag, subdiag)
            erreur_diff = np.abs((E_num_diff[:10] - E_ref) / np.abs(E_ref))
            erreurs_diff_finies.append(erreur_diff)
            
            diag_n, subdiag_n = construire_hamiltonien_numerov(N, x, potentiel)
            E_num_numerov, _ = calculer_etats_propres(diag_n, subdiag_n)
            erreur_numerov = np.abs((E_num_numerov[:10] - E_ref) / np.abs(E_ref))
            erreurs_numerov.append(erreur_numerov)
            
            print(f"N = {N:>4} | Erreur moyenne (Diff. finies): {np.mean(erreur_diff):.3e} | "
                  f"Erreur moyenne (Numerov): {np.mean(erreur_numerov):.3e}")

        erreurs_diff_finies = np.array(erreurs_diff_finies)
        erreurs_numerov = np.array(erreurs_numerov)
        
        plt.figure(figsize=(12, 8))
        couleurs = plt.cm.plasma(np.linspace(0, 1, 10))
        for k in range(10):
            plt.plot(Ns, erreurs_diff_finies[:, k], '-o', color=couleurs[k], label=f'État {k} (Diff. finies)', alpha=0.7)
            plt.plot(Ns, erreurs_numerov[:, k], '--s', color=couleurs[k], label=f'État {k} (Numerov)', alpha=0.7)
        plt.title(f"Convergence des énergies numériques\n({nom})")
        plt.xlabel("Nombre de points N")
        plt.ylabel("Erreur relative")
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.show()

def analyse_erreur_tous_potentiels_largeur():
    """
    Analyse de l'erreur pour tous les potentiels en variant la largeur de la boîte L,
    pour un nombre fixe de points N=500, pour les 10 premiers états.
    
    Paramètres:
    - N: nombre de points de discrétisation fixé à 500
    - Ls: liste des largeurs de boîte à tester (spécifique à chaque potentiel)
    - n_states: nombre d'états quantiques à considérer (fixé à 10)
    
    Visualisation:
    - Un graphique par potentiel, montrant l'erreur relative en fonction de L
    - Même couleur par état, 'o' pour différences finies, 's' pour Numerov
    """
    N = 500
    n_states = 10
    N_ref = 2000
    
    potentiels = [
        ("Oscillateur Harmonique", lambda x: potentiel_harmonique(x, omega=1.0), [6, 8, 10, 12, 14, 16], True),
        ("Puits Fini", lambda x: potentiel_puits(x, V0=50, largeur=3), [6, 8, 10, 12, 14, 16], False),
        ("Barrière", lambda x: potentiel_barriere(x, V0=50, largeur=2), [10, 12, 14, 16, 18, 20], False),
        ("Double Puits", lambda x: potentiel_double_puits(x, V0=50, largeur_puits=1, distance=4), [10, 12, 14, 16, 18, 20], False),
        ("Hydrogène Radial (l=0)", lambda r: potentiel_hydrogene_radial(r, l=0), [30, 40, 50, 60, 70, 80], True)
    ]

    for nom, potentiel, Ls, theorie in potentiels:
        print(f"\nAnalyse de l'erreur pour {nom} en variant la largeur de la boîte (N={N}):")
        
        # Solution de référence
        L_ref = max(Ls)
        x_ref = np.linspace(1e-6 if nom == "Hydrogène Radial (l=0)" else -L_ref/2, 
                           L_ref if nom == "Hydrogène Radial (l=0)" else L_ref/2, N_ref)
        if theorie:
            if nom == "Oscillateur Harmonique":
                E_ref = np.array([hbar * omega * (n + 0.5) for n in range(n_states)])
            else:
                E_ref = np.array([-0.5 / (n + 1)**2 for n in range(n_states)])
        else:
            diag_ref, subdiag_ref = construire_hamiltonien(N_ref, x_ref, potentiel)
            E_ref, _ = calculer_etats_propres(diag_ref, subdiag_ref)
            E_ref = E_ref[:n_states]
        
        erreurs_diff_finies = []
        erreurs_numerov = []
        
        for L in Ls:
            x = np.linspace(1e-6 if nom == "Hydrogène Radial (l=0)" else -L/2, 
                           L if nom == "Hydrogène Radial (l=0)" else L/2, N)
            
            # Différences finies
            diag, subdiag = construire_hamiltonien(N, x, potentiel)
            E_num_diff, _ = calculer_etats_propres(diag, subdiag)
            erreur_diff = np.abs((E_num_diff[:n_states] - E_ref) / np.abs(E_ref))
            erreurs_diff_finies.append(erreur_diff)
            
            # Numerov
            diag_n, subdiag_n = construire_hamiltonien_numerov(N, x, potentiel)
            E_num_numerov, _ = calculer_etats_propres(diag_n, subdiag_n)
            erreur_numerov = np.abs((E_num_numerov[:n_states] - E_ref) / np.abs(E_ref))
            erreurs_numerov.append(erreur_numerov)
            
            print(f"L = {L:>2} | Erreur moyenne (Diff finies): {np.mean(erreur_diff):.3e} | "
                  f"Erreur moyenne (Numerov): {np.mean(erreur_numerov):.3e}")
        
        erreurs_diff_finies = np.array(erreurs_diff_finies)
        erreurs_numerov = np.array(erreurs_numerov)
        
        plt.figure(figsize=(10, 6))
        couleurs = plt.cm.plasma(np.linspace(0, 1, n_states))
        for k in range(n_states):
            plt.plot(Ls, erreurs_diff_finies[:, k], '-o', color=couleurs[k], 
                     label=f'État {k} (Diff. finies)', alpha=0.7)
            plt.plot(Ls, erreurs_numerov[:, k], '--s', color=couleurs[k], 
                     label=f'État {k} (Numerov)', alpha=0.7)
        plt.title(f"Convergence des énergies numériques\n({nom}, N={N})")
        plt.xlabel("Largeur de la boîte L")
        plt.ylabel("Erreur relative")
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.show()

# ======================================================================
# INTERFACE UTILISATEUR COMPLÈTE
# ======================================================================
def afficher_menu():
    """Affiche le menu principal avec les options disponibles"""
    print("\n" + "="*70)
    print(" SIMULATEUR QUANTIQUE 1D - MENU PRINCIPAL ".center(70))
    print("="*70)
    print("Options disponibles:")

options = {
    '1': ("Oscillateur Harmonique", etude_oscillateur_harmonique),
    '2': ("Puits de Potentiel Fini", etude_puits_fini),
    '3': ("Barrière de Potentiel", etude_barriere),
    '4': ("Double Puits Quantique", etude_double_puits),
    '5': ("Atome d'Hydrogène Radial (l=0)", etude_hydrogene_radial),
    '6': ("Analyse d'erreur (tous les potentiels)", analyse_erreur_tous_potentiels),
    '7': ("Analyse d'erreur (tous potentiels, variation de L)", analyse_erreur_tous_potentiels_largeur),
    '8': ("Toutes les études", None),
    'Q': ("Quitter", None)
}

def menu_principal():
    """Boucle principale du programme"""
    while True:
        afficher_menu()
        for key, (description, _) in options.items():
            print(f"  {key}. {description}")
        choix = input("\nVotre choix: ").strip().upper()
        if choix == 'Q':
            print("\nMerci d'avoir utilisé le simulateur quantique !")
            break
        elif choix == '8':
            print("\nExécution de toutes les études...")
            for key in options:
                if key not in ['8', 'Q'] and options[key][1] is not None:
                    print(f"\n>> {options[key][0]} <<")
                    options[key][1]()
                    if key not in ['5', '6', '7']:
                        input("\nAppuyez sur Entrée pour continuer...")
        elif choix in options and options[choix][1] is not None:
            print(f"\n>> {options[choix][0]} <<")
            options[choix][1]()
            if choix not in ['5', '6', '7']:
                input("\nAppuyez sur Entrée pour continuer...")
        else:
            print("Option invalide. Veuillez choisir une option valide.")

# ======================================================================
# POINT D'ENTRÉE PRINCIPAL
# ======================================================================
if __name__ == "__main__":
    print("""
    SIMULATION QUANTIQUE 1D
    -----------------------
    Ce programme résout numériquement l'équation de Schrödinger
    pour différents potentiels quantiques en une dimension.
    """)
    menu_principal()

