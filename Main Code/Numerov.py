
import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
hbar = 1.0  # Constante de Planck réduite
m = 1.0     # Masse de la particule
omega = 1.0 # Fréquence angulaire de l'oscillateur
L = 8.0     # Limite de la grille spatiale [-L, L]
N = 2000    # Nombre de points de grille
dx = 2 * L / (N - 1)  # Pas spatial
x = np.linspace(-L, L, N)  # Grille spatiale

# Potentiel de l'oscillateur harmonique : V(x) = (1/2) * m * omega^2 * x^2
def potentiel(x):
    return 0.5 * m * omega**2 * x**2

# Algorithme de Numerov pour résoudre l'équation de Schrödinger
def numerov(E, x, dx):
    psi = np.zeros(N)  # Tableau pour la fonction d'onde
    psi[0] = 0.0       # Condition aux limites : ψ(-L) = 0
    psi[1] = 1e-6    # Petite valeur initiale pour ψ au deuxième point

    k2 = 2 * m * (E - potentiel(x)) / hbar**2

    for i in range(1, N-1):
        psi[i+1] = (2 * psi[i] * (1 - (5/12) * dx**2 * k2[i]) - 
                    psi[i-1] * (1 + (1/12) * dx**2 * k2[i-1])) / \
                   (1 + (1/12) * dx**2 * k2[i+1])
    
    return psi

# Recherche des valeurs propres d'énergie par dichotomie
def trouver_valeur_propre(E_min, E_max, tol=1e-15, max_iter=1000):
    for i in range(max_iter):
        E_mid = (E_min + E_max) / 2
        psi_mid = numerov(E_mid, x, dx)
        valeur_bord = psi_mid[-1]

        psi_min = numerov(E_min, x, dx)
        
        if abs(valeur_bord) < tol:
            return E_mid, psi_mid
        elif valeur_bord * psi_min[-1] < 0:
            E_max = E_mid
        else:
            E_min = E_mid

        if E_max - E_min < tol:
            return E_mid, psi_mid

    raise ValueError(f"Convergence non atteinte pour E_min={E_min}, E_max={E_max}")

# Calcul des premiers niveaux d'énergie
n_niveaux = 3
energies = []
fonctions_onde = []

for n in range(n_niveaux):
    E_min = (n - 0.1) * hbar * omega
    E_max = (n + 1.1) * hbar * omega
    E, psi = trouver_valeur_propre(E_min, E_max)
    
    norme = np.sqrt(np.sum(psi**2) * dx)
    psi = psi / norme if norme != 0 else psi
    
    energies.append(E)
    fonctions_onde.append(psi)

# Affichage des résultats
for n, E in enumerate(energies):
    print(f"Niveau n={n}: Énergie = {E:.4f} (Analytique: {(n + 0.5) * hbar * omega:.4f})")

# Visualisation des fonctions d'onde
plt.figure(figsize=(10, 6))
for n, psi in enumerate(fonctions_onde):
    plt.plot(x, psi, label=f'n={n}')
plt.title("Fonctions d'onde de l'oscillateur harmonique quantique")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.grid(True)
plt.show()