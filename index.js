import express from 'express';
import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline, layer_norm } from '@xenova/transformers';
import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from 'dotenv';
import fs from 'fs';

// Cargar las variables de entorno desde el archivo .env
dotenv.config();

// Inicializa Pinecone y el modelo de embeddings
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const app = express();
const port = 4000;

// Middleware para parsear las solicitudes JSON
app.use(express.json());

// Contador de preguntas
let question_count = 0;

// Función para registrar las preguntas y respuestas en un archivo de log
const chatResponses = (question, response) => {
  const logEntry = `Pregunta: ${question}\nRespuesta: ${response}\n\n`;
  fs.appendFileSync('chat_responses.log', logEntry, 'utf8');
};

// Función para realizar la consulta a Pinecone y generar respuesta con Gemini
async function consultarYGenerarRespuesta(query) {
  const indexName = "negocios"; // Cambia al nombre correcto de tu índice
  const indices = await pc.listIndexes();

  // Verifica si el índice existe
  if (Array.isArray(indices.indexes) && indices.indexes.some(index => index.name === indexName)) {
    // Extrae embeddings para la consulta
    const extractor = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5');
    let embeddings = await extractor([query], { pooling: 'mean' });

    // Ajusta la dimensión esperada para el índice
    const matryoshka_dim = 384; // Asegúrate que esta dimensión coincida con tu índice en Pinecone
    embeddings = layer_norm(embeddings, [embeddings.dims[1]])
      .slice(null, [0, matryoshka_dim])
      .normalize(2, -1)
      .tolist();

    // Realiza la consulta de similitud en Pinecone
    const index = pc.Index(indexName);
    const resultado = await index.query({
      vector: embeddings[0],
      topK: 8,  // Número de resultados deseados
      includeMetadata: true
    });

    // Extrae los fragmentos relevantes de Pinecone
    const contextoPinecone = resultado.matches
      .map(match => match.metadata.text)
      .join("\n")
      .slice(0, 10000); // Limita la longitud del contexto a 10000 caracteres

    if (!contextoPinecone) {
      return "Lo siento, no pude encontrar información relevante para tu consulta en la base de datos.";
    }

    // Estructura de prompt para la pregunta
    const prompt = `
    Eres un agente de servicio de un banco con experiencia en la gestión de reclamos.  
    Tu tarea es **clasificar la siguiente consulta** en una de las siguientes categorías:  
    
    1️⃣ **Reclamo Operativo**: Problemas con transacciones, cargos indebidos, reembolsos, bloqueo de cuentas, demoras en procesos.  
    2️⃣ **Reclamo Técnico**: Fallos en la app, error en cajeros, problemas con banca en línea, accesos denegados, errores de autenticación.  
    3️⃣ **Reclamo Atención al Cliente**: Mala atención en sucursales, trato inadecuado de asesores, tiempos de espera largos, falta de respuesta.  
    4️⃣ **Otro**: Solo si la pregunta no tiene suficiente contexto o no se relaciona con el banco.  
    
    📌 **Ejemplo 1:** "No puedo acceder a mi cuenta en la app." → Reclamo Técnico  
    📌 **Ejemplo 2:** "Me cobraron dos veces el mismo pago." → Reclamo Operativo  
    📌 **Ejemplo 3:** "El asesor del banco me trató mal." → Reclamo Atención al Cliente  
    📌 **Ejemplo 4:** "¿Cuándo es el próximo partido de fútbol?" → Otro  
    
    Aquí tienes información relevante de la base de datos que puede ayudar a clasificar:  
    📝 **Contexto encontrado:**  
    ${contextoPinecone}  
    
    🔹 **Clasifica la siguiente pregunta:**  
    ❓ **Pregunta:** ${query}  
    
    ✏️ **Devuelve solo la categoría exacta (sin explicaciones adicionales)**.
    `;
    
    // Crear una instancia de GoogleGenerativeAI con la API Key desde las variables de entorno
    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    const model = genAI.getGenerativeModel({
      model: "gemini-1.0-pro",
    });

    // Iniciar la sesión de chat con el modelo y configuración
    const generationConfig = {
      temperature: 1,            // Controla la aleatoriedad de la respuesta
      topP: 0.95,               // Controla el recorte de la probabilidad acumulada
      topK: 40,                 // Controla cuántos posibles tokens usar para la generación
      maxOutputTokens: 8192,    // Número máximo de tokens de salida
      responseMimeType: "text/plain", // Tipo MIME de la respuesta
    };

    const chatSession = model.startChat({
      generationConfig,
      history: [], // No historial en este caso
    });

    // Enviar el mensaje al modelo Gemini
    const result = await chatSession.sendMessage(prompt);

    // Extraer solo la respuesta esperada
    const respuesta = result.response.text().trim();

    if (!respuesta) {
      return "Lo siento, no pude generar una respuesta adecuada para tu consulta.";
    }

    // Registra la pregunta y la respuesta en el archivo de registro
    chatResponses(query, respuesta);

    return respuesta; // Devuelve solo la respuesta generada sin la frase adicional
  } else {
    return "Lo siento, no pudimos encontrar el índice para realizar la consulta.";
  }
}

// Endpoint para recibir la consulta y responder
app.post('/chat', async (req, res) => {
  const { question } = req.body; // Obtiene la pregunta del cuerpo de la solicitud

  if (!question) {
    return res.status(400).json({ error: 'No se proporcionó ninguna pregunta.' });
  }

  try {
    question_count += 1; // Incrementar contador de preguntas

    const respuesta = await consultarYGenerarRespuesta(question);
    res.json({
      data: respuesta,
      message: "La petición se procesó correctamente",
      question_count: question_count, // Incluir el contador de preguntas en la respuesta
    });
  } catch (error) {
    console.error("Error en la consulta:", error);
    res.status(500).json({ error: 'Ocurrió un error al procesar tu solicitud.' });
  }
});

// Inicia el servidor en el puerto 3000
app.listen(port, () => {
  console.log(`Servidor corriendo en http://localhost:${port}`);
});
