package dev.flutterberlin.flutter_gemma

import android.content.Context
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.*

/** FlutterGemmaPlugin */
class FlutterGemmaPlugin : FlutterPlugin, MethodChannel.MethodCallHandler, EventChannel.StreamHandler {
  private lateinit var channel: MethodChannel
  private lateinit var eventChannel: EventChannel
  private var eventSink: EventChannel.EventSink? = null
  private lateinit var inferenceModel: InferenceModel
  private lateinit var context: Context
//  private val scope = CoroutineScope(Dispatchers.IO) // Use IO for background tasks

  private val lowPriorityDispatcher = Dispatchers.IO.limitedParallelism(1)
  private val scope = CoroutineScope(
    lowPriorityDispatcher +
            Job() +
            CoroutineExceptionHandler { _, throwable ->
              println("Inference coroutine error: ${throwable.message}")
            }
  )


  override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    context = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_gemma")
    channel.setMethodCallHandler(this)
    eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, "flutter_gemma_stream")
    eventChannel.setStreamHandler(this)
  }

  override fun onMethodCall(call: MethodCall, result: Result) {
    when (call.method) {
      "init" -> {
        // Initialization code...
        try {
          val modelPath = call.argument<String>("modelPath")!!
          val maxTokens = call.argument<Int>("maxTokens")!!
          val temperature = call.argument<Float>("temperature")!!
          val randomSeed = call.argument<Int>("randomSeed")!! // Fix argument name
          val topK = call.argument<Int>("topK")!!
          val loraPath = call.argument<String?>("loraPath")
          val numOfSupportedLoraRanks = call.argument<Int?>("numOfSupportedLoraRanks")
          val supportedLoraRanks = call.argument<List<Int>?>("supportedLoraRanks")

          inferenceModel = InferenceModel.getInstance(
            context, modelPath, maxTokens, temperature,
            randomSeed, topK, loraPath, numOfSupportedLoraRanks, supportedLoraRanks
          )
          result.success(true)
        } catch (e: Exception) {
          result.error("ERROR", "Failed to initialize gemma", e.localizedMessage)
        }
      }
      "getGemmaResponse" -> {
        // Synchronous response generation...
      }
//      "getGemmaResponseAsync" -> {
//        val prompt = call.argument<String>("prompt")!!
//        scope.launch { // Launch in IO dispatcher
//          try {
//            val answer = inferenceModel.generateResponseAsync(prompt)
//            // Switch back to Main to send the result
//            withContext(Dispatchers.Main) {
//              result.success(answer)
//            }
//          } catch (e: Exception) {
//            // Switch back to Main for error handling
//            withContext(Dispatchers.Main) {
//              result.error("ERROR", "Failed to get async gemma response", e.localizedMessage)
//            }
//          }
//        }
//      }
      "getGemmaResponseAsync" -> {
        val prompt = call.argument<String>("prompt")!!
        scope.launch {
          // Android-specific thread priority adjustment
          android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_LOWEST)


          // Add thread priority adjustment
          Thread.currentThread().apply {
            priority = Thread.MIN_PRIORITY
          }

          try {
            val answer = inferenceModel.generateResponseAsync(prompt)
            withContext(Dispatchers.Main) {
              result.success(answer)
            }
          } catch (e: Exception) {
            withContext(Dispatchers.Main) {
              result.error("ERROR", "Failed to get async gemma response", e.localizedMessage)
            }
          }
        }
      }
      else -> result.notImplemented()
    }
  }

  override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
    eventSink = events
    scope.launch {
      // Android-specific thread priority adjustment
      android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_LOWEST)

      inferenceModel.partialResults.collect { pair ->
        withContext(Dispatchers.Main) { // Ensure we send events on the Main thread
          if (pair.second) {
            events?.success(pair.first)
            events?.success(null)
          } else {
            events?.success(pair.first)
          }
        }
      }
    }
  }

  override fun onCancel(arguments: Any?) {
    eventSink = null
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
    eventChannel.setStreamHandler(null)
  }
}
