// lib/core/globals.dart
// Updated & fully compliant with your FastAPI backend
// Added: ringRoadSimulation endpoint
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

class AppConfig {
  static String get baseUrl {
    if (kIsWeb) {
      return 'http://localhost:8000';
    }
    if (Platform.isAndroid) {
      return 'http://10.0.2.2:8000'; // Android emulator → localhost
    }
    if (Platform.isIOS) {
      return 'http://localhost:8000'; // iOS simulator → localhost
    }
    // Fallback (physical device, Linux/desktop, etc.)
    return 'http://localhost:8000';
  }

  static String get login => '$baseUrl/login';
  static String get register => '$baseUrl/register';
  static String get me => '$baseUrl/me';
  static String get adminRequestVerification => '$baseUrl/admin/request-verification';
  static String get adminCompleteRegistration => '$baseUrl/admin/complete-registration';
  static String checkAdminVerification(String email) => '$baseUrl/check-admin-verification/$email';
  static String get findStops => '$baseUrl/find-stops';
  static String get saveRoute => '$baseUrl/admin/save-route';
  static String get routes => '$baseUrl/routes';
  static String routeDetail(int id) => '$baseUrl/route/$id';
  static String updateRoute(int id) => '$baseUrl/admin/update-route/$id';
  static String deleteRoute(int id) => '$baseUrl/admin/delete-route/$id';
  static String get changePassword => '$baseUrl/change-password';
  static String get deleteAccount => '$baseUrl/delete-account';
  static String get buses => '$baseUrl/buses';
  static String updateBus(int id) => '$baseUrl/admin/update-bus/$id';
  static String deleteBus(int id) => '$baseUrl/admin/delete-bus/$id';
  static String get predictEta => '$baseUrl/predict-eta';

  // Existing correct map route
  static String mapRoute(int routeId) => '$baseUrl/map/route/$routeId';

  // NEW: Ring Road simulation endpoint
  static String get ringRoadSimulation => '$baseUrl/map/simulate/ring-road';
}

/// Quick platform helpers
bool get isWeb => kIsWeb;
bool get isAndroid => !kIsWeb && Platform.isAndroid;
bool get isIOS => !kIsWeb && Platform.isIOS;
bool get isMobile => isAndroid || isIOS;