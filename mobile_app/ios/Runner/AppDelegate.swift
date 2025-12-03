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

    // --- CORE ML SETUP ---

    // Get the Flutter Controller
    let controller : FlutterViewController = window?.rootViewController as! FlutterViewController

    // Create the Channel
    let pbrChannel = FlutterMethodChannel(name: "com.NeuroPBR/pbr_generator",
      binaryMessenger: controller.binaryMessenger)

    // Init the Model (Safely check for iOS 15)
    if #available(iOS 15.0, *) {
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

      // Validate Arguments
      guard let args = call.arguments as? [String: Any],
      let v1 = args["view1"] as? FlutterStandardTypedData,
      let v2 = args["view2"] as? FlutterStandardTypedData,
      let v3 = args["view3"] as? FlutterStandardTypedData else {
        result(FlutterError(code: "INVALID_ARGS", message: "Expected view1, view2, view3 as bytes", details: nil))
        return
      }

      // Run Model
      if #available(iOS 15.0, *) {
        DispatchQueue.global(qos: .userInitiated).async {
          do {
            let outputs = try self?.pbrHandler?.generateMaps(
              view1Data: v1.data,
              view2Data: v2.data,
              view3Data: v3.data
            )
            DispatchQueue.main.async {
              result(outputs)
            }
          } catch {
            DispatchQueue.main.async {
              result(FlutterError(code: "MODEL_ERROR", message: error.localizedDescription, details: nil))
            }
          }
        }
      } else {
        result(FlutterError(code: "OS_OBSOLETE", message: "Requires iOS 15+", details: nil))
      }
    })

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}