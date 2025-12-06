import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  //Keep a persistent instance of the model handler
  private var pbrHandler: PBRModelHandler?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)

    if let registrar = self.registrar(forPlugin: "NeuropbrMetalRendererPlugin") {
      NeuropbrMetalRendererPlugin.register(with: registrar)
    }
    
    // --- IMAGE PROCESSOR SETUP ---
    if let imageProcessorRegistrar = self.registrar(forPlugin: "ImageProcessorPlugin") {
      ImageProcessorPlugin.register(with: imageProcessorRegistrar)
    }

    // --- CORE ML SETUP ---

    // Get the Flutter Controller
    let controller : FlutterViewController = window?.rootViewController as! FlutterViewController

    // Create the Channel
    let pbrChannel = FlutterMethodChannel(name: "com.NeuroPBR/pbr_generator",
      binaryMessenger: controller.binaryMessenger)

    // Init the Model (Safely check for iOS 17)
    if #available(iOS 17.0, *) {
      print("Initializing PBR Model...")
      self.pbrHandler = PBRModelHandler()
    }

    // Handle Method Calls
    pbrChannel.setMethodCallHandler({
      [weak self] (call: FlutterMethodCall, result: @escaping FlutterResult) -> Void in

      guard call.method == "generatePBR" else {
        result(FlutterMethodNotImplemented)
        return
      }

      if #available(iOS 17.0, *) {
        // Delegate to the handler which parses args and runs the model
        self?.pbrHandler?.generatePBR(call: call, result: result)
      } else {
        result(FlutterError(code: "OS_OBSOLETE", message: "Requires iOS 17+", details: nil))
      }
    })

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}