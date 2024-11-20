import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo CSV
data = pd.read_csv("resultados_tiempos_y_eficiencia.csv")

# Definir las posiciones de las barras
x = np.arange(len(data["Ejecución"]))

# Gráfico de tiempos de ejecución
plt.figure(figsize=(10, 6))
bar_width = 0.25

# Crear barras
plt.bar(x - bar_width, data["Tiempos_Base_ms"], width=bar_width, label="Global")
plt.bar(x, data["Tiempos_Const_ms"], width=bar_width, label="Constante")
plt.bar(x + bar_width, data["Tiempos_Shared_ms"], width=bar_width, label="Compartida")

# Configuración del gráfico
plt.title("Tiempos de Ejecución por Enfoque")
plt.xlabel("Ejecución")
plt.ylabel("Tiempo (ms)")
plt.xticks(x, data["Ejecución"], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("grafico_tiempos_barras.png")  # Guardar la gráfica como archivo
plt.show()

# Gráfico de speedup
plt.figure(figsize=(10, 6))

# Crear barras
plt.bar(x - bar_width, data["Speedup_Base_Const"], width=bar_width, label="Global vs Constante")
plt.bar(x, data["Speedup_Const_Shared"], width=bar_width, label="Constante vs Compartida")
plt.bar(x + bar_width, data["Speedup_Base_Shared"], width=bar_width, label="Global vs Compartida")

# Configuración del gráfico
plt.title("Speedup por Comparación")
plt.xlabel("Ejecución")
plt.ylabel("Speedup")
plt.xticks(x, data["Ejecución"], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("grafico_speedup_barras.png")  # Guardar la gráfica como archivo
plt.show()
