import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Funcion de Visualizacion de Analisis Univariable - Numerico

def uninum(df):
    num = df.select_dtypes(include='number').columns
    vars = len(num)
    
    fig, axes = plt.subplots(nrows=vars, ncols=2, figsize=(15, 5 * vars))
    fig.subplots_adjust(hspace=0.4)

    for i, col in enumerate(num):
        #1
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color='darkblue')
        axes[i, 0].set_title(f'Distribución de {col}', fontsize=12)
        
        #2
        sns.boxplot(x=df[col], ax=axes[i, 1], color='green')
        axes[i, 1].set_title(f'Outliers de {col}', fontsize=12)

    plt.tight_layout()
    plt.show()


#Funcion de Visualizacion de Analisis Univariable - Categorico

def unicat(df):
    cat = df.select_dtypes(include='str').columns
    vars = len(cat)

    fig, axes = plt.subplots(nrows=vars, ncols=1, figsize=(10,5*vars))
    fig.subplots_adjust(hspace=0.4)

    for i, col in enumerate(cat):

        sns.histplot(df[col], kde=True, ax=axes[i], color='red')
        axes[i].set_title(f'Distribución de {col}', fontsize=12)

    plt.tight_layout()
    plt.show()

#Funcion de Visualizacion de Analisis Bivariante - Numerico

def binum(df, target):
    # 1. Seleccionamos las columnas numéricas (excluyendo el target para las X)
    num = df.select_dtypes(include=['number']).columns
    cols = [c for c in num if c != target]
    n_vars = len(cols)

    # 2. Creamos la cuadrícula: 2 filas y N columnas (una por variable)
    # El tamaño se ajusta según la cantidad de variables
    fig, axes = plt.subplots(nrows=2, ncols=n_vars, figsize=(5 * n_vars, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, col in enumerate(cols):
        # --- FILA 1: Regplot (Dispersión + Línea de tendencia) ---
        sns.regplot(data=df, x=col, y=target, ax=axes[0, i], 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
        axes[0, i].set_title(f'{target} vs {col}', fontsize=12)
        
        # --- FILA 2: Matriz de Correlación Individual ---
        # Calculamos la correlación solo entre la variable actual y el target
       
        
        sns.heatmap(df[[col, target]].corr(), annot=True, fmt='.2f', 
                    cbar=False, ax=axes[1, i], square=True)
        axes[1, i].set_title(f'Correlacion: {col}', fontsize=10)

    plt.tight_layout()
    plt.show()

#Funcion de Visualizacion de Analisis Bivariante - Categorico

def bicat(df, target='y'):

    cat = df.select_dtypes(include=['str']).columns
    cat = [col for col in cat if col != target]
    
    vars = len(cat)
    
    # 2. Configuramos la cuadrícula: 1 columna por cada variable
    # Usamos un tamaño dinámico para que no se vea todo amontonado
    fig, axes = plt.subplots(nrows=vars, ncols=1, figsize=(12, 6 *vars))
    fig.subplots_adjust(hspace=0.4)

    for i, col in enumerate(cat):
        
        sns.countplot(data=df, x=col, hue=target, ax=axes[i], palette='viridis', 
                      order=df[col].value_counts().index, dodge=True)
        
        axes[i].set_title(f'Impacto de {col.upper()} en la contratación del depósito', fontsize=14)
        axes[i].set_xlabel(col.capitalize(), fontsize=12)
        axes[i].set_ylabel('Cantidad de Clientes', fontsize=12)
        axes[i].legend(title='¿Contrató?', loc='upper right')
        
        # Rotamos las etiquetas del eje X si hay muchos nombres (como en 'job')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

#Funcion de Visualizacion de Analisis Multivariante Num-Cat

def multiV(dfnum): #Incluir en un solo DF la variable objetivo

    plt.figure(figsize=(20,20))
    sns.heatmap(dfnum.corr(), annot= True, fmt='.2f', cmap='viridis')

    plt.tight_layout()
    plt.show()
