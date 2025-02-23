# Usa una imagen oficial de Node.js
FROM node:18-alpine

# Instalar las dependencias necesarias para que las bibliotecas de Node.js puedan ser cargadas correctamente
RUN apk update && apk add --no-cache \
    libc6-compat  # Esto instalará glibc en la imagen Alpine

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY package*.json ./

# Instala las dependencias de Node.js
RUN npm install

# Copia el resto de los archivos
COPY . .

# Expone el puerto en el que corre el backend
EXPOSE 4000

# Comando para iniciar el backend
CMD ["node", "index.js"]
