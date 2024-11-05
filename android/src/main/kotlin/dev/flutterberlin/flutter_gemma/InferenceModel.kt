package dev.flutterberlin.flutter_gemma

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.io.File
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CompletableDeferred

class InferenceModel private constructor(
    context: Context,
    private val modelPath: String,
    maxTokens: Int,
    temperature: Float,
    randomSeed: Int,
    topK: Int,
    loraPath: String?,
    numOfSupportedLoraRanks: Int?,
    supportedLoraRanks: List<Int>?,
) {
    private var llmInference: LlmInference

    private val _partialResults = MutableSharedFlow<Pair<String, Boolean>>(
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val partialResults: SharedFlow<Pair<String, Boolean>> = _partialResults.asSharedFlow()

    private val modelExists: Boolean
        get() = File(modelPath).exists()

    init {
        if (!modelExists) {
            throw IllegalArgumentException("Model not found at path: $modelPath")
        }
        val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(maxTokens)
            .setTemperature(temperature)
            .setRandomSeed(randomSeed)
            .setTopK(topK)
            .setResultListener { partialResult, done ->
                _partialResults.tryEmit(partialResult to done)
            }

        numOfSupportedLoraRanks?.let { optionsBuilder.setNumOfSupportedLoraRanks(it) }
        supportedLoraRanks?.let { optionsBuilder.setSupportedLoraRanks(it) }
        loraPath?.let { optionsBuilder.setLoraPath(it) }

        val options = optionsBuilder.build()

        llmInference = try {
            LlmInference.createFromOptions(context, options)
        } catch (e: Exception) {
            throw RuntimeException("Failed to create LlmInference instance: ${e.message}", e)
        }
    }

    fun generateResponse(prompt: String): String? {
        return llmInference.generateResponse(prompt)
    }

    suspend fun generateResponseAsync(prompt: String): String? {
        // Use a CompletableDeferred to wait for the result.
        val completion = CompletableDeferred<String?>()

        withContext(Dispatchers.IO) {
            // This assumes the method does not return a result directly.
            llmInference.generateResponseAsync(prompt) // Call without expecting a return value.
            // You may need to set up your result listener here if necessary.
            // For example, if you have an interface to handle results.
        }

        return completion.await() // Await completion, may need to change this based on actual use case.
    }

    companion object {
        @Volatile
        private var instance: InferenceModel? = null

        fun getInstance(
            context: Context,
            modelPath: String,
            maxTokens: Int,
            temperature: Float,
            randomSeed: Int,
            topK: Int,
            loraPath: String?,
            numOfSupportedLoraRanks: Int?,
            supportedLoraRanks: List<Int>?
        ): InferenceModel {
            return instance ?: synchronized(this) {
                instance ?: InferenceModel(
                    context,
                    modelPath,
                    maxTokens,
                    temperature,
                    randomSeed,
                    topK,
                    loraPath,
                    numOfSupportedLoraRanks,
                    supportedLoraRanks
                ).also { instance = it }
            }
        }
    }
}
