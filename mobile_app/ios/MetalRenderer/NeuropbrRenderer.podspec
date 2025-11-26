Pod::Spec.new do |s|
  s.name             = 'NeuropbrRenderer'
  s.version          = '0.0.1'
  s.summary          = 'A Metal-based PBR renderer for NeuroPBR.'
  s.description      = <<-DESC
A custom Metal renderer plugin for Flutter, handling PBR rendering, environment maps, and material previews.
                       DESC
  s.homepage         = 'https://josephhelfenbein.com'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Joseph Helfenbein' => 'josephhelfenbein@gmail.com' }
  s.source           = { :path => '.' }
  s.source_files = '**/*.{h,m,mm,cpp,hpp,metal}'
  s.public_header_files = 'NeuropbrMetalRendererPlugin.h'
  s.platform = :ios, '13.0'
  
  # Flutter dependencies
  s.dependency 'Flutter'
  
  s.frameworks = 'Metal', 'MetalKit', 'MobileCoreServices'
  
  # Build settings
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'GCC_PREPROCESSOR_DEFINITIONS' => 'GL_SILENCE_DEPRECATION',
    'DEFINES_MODULE' => 'YES',
  }
end
