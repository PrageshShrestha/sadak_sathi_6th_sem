// lib/screens/map_screen.dart
// FULLY UPDATED – Real-time server sync every second
// Buses & stops in bottom panel are clickable (tap to center on map)
// UI refined with Shneiderman's 8 Golden Rules:
// 1. Consistency – uniform colors, icons, typography
// 2. Shortcuts – direct tap-to-focus
// 3. Feedback – subtle elevation & ripple on taps
// 4. Closure – clear "From → Next" flow
// 5. Error prevention – safe defaults & null checks
// 6. Reversible actions – panel expand/collapse
// 7. User control – full interaction on buses/stops
// 8. Reduce memory load – concise, hierarchical info

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_cancellable_tile_provider/flutter_map_cancellable_tile_provider.dart';
import 'package:latlong2/latlong.dart' as latlng;
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../widgets/filter_panel.dart';
import '../screens/auth_screen.dart';
import 'setting_screen.dart';
import '../services/api_service.dart';
import 'package:url_launcher/url_launcher.dart';
import 'admin_feature_page.dart';
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../core/globals.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen>
    with SingleTickerProviderStateMixin {
  final MapController _mapController = MapController();
  final storage = const FlutterSecureStorage();
  latlng.LatLng _currentLocation = const latlng.LatLng(27.7172, 85.3240);
  bool _isLoadingLocation = true;
  String? _userName;
  bool _isAdmin = false;
  late final AnimationController _fabController;
  late final Animation<double> _fabScale;

  List<dynamic> _availableRoutes = [];
  dynamic _selectedRoute;
  List<dynamic> _busesOnRoute = [];
  List<latlng.LatLng> _routePoints = [];
  List<dynamic> _stops = [];
  List<SimulatedBus> _simulatedBuses = [];

  Timer? _updateTimer;
  dynamic _selectedItem;
  bool _showPopup = false;

  // Expandable bus list panel
  bool _isBusListExpanded = false;

  final List<String> stopsOrder = [
    "koteshwor",
    "airport",
    "gausala",
    "chabhil",
    "gangabu",
    "samakhushi",
    "balaju",
    "banasthali",
    "swoyambhu",
    "satdobato",
    "gwarko",
    "balkumari",
  ];

  final Map<String, List<double>> busStations = {
    "koteshwor": [
      27.679261426232177,
      85.34936846935956,
      27.680728503597233,
      85.3495489426487,
    ],
    "airport": [
      27.70078540318246,
      85.3533769530794,
      27.701573335528412,
      85.35303133213452,
    ],
    "gausala": [
      27.706379161514995,
      85.34493560000823,
      27.70725888643062,
      85.34418441639279,
    ],
    "chabhil": [
      27.717031239951016,
      85.3463460556796,
      27.718204535772227,
      85.34685091098906,
    ],
    "gangabu": [
      27.73756021680712,
      85.32456480569198,
      27.738041319788685,
      85.32514445895883,
    ],
    "samakhushi": [
      27.734510272551045,
      85.31315704526047,
      27.734988742644767,
      85.31546713620789,
    ],
    "balaju": [
      27.726016065272873,
      85.3035017969009,
      27.728129139328153,
      85.30523648797792,
    ],
    "banasthali": [
      27.71883550527814,
      85.2858452410448,
      27.719844381320847,
      85.28726472663818,
    ],
    "swoyambhu": [
      27.71540016240659,
      85.28353069074375,
      27.716986341444347,
      85.28386474803716,
    ],
    "satdobato": [
      27.658031146589305,
      85.32347613583907,
      27.659581479957513,
      85.32579715989439,
    ],
    "gwarko": [
      27.666433507619914,
      85.33196568965492,
      27.667210717280035,
      85.33248571687696,
    ],
    "balkumari": [
      27.671076985194357,
      85.33969391131329,
      27.672139477723608,
      85.34068226337071,
    ],
  };

  final Map<String, latlng.LatLng> stopCenters = {
    "koteshwor": latlng.LatLng(27.6799, 85.3494),
    "airport": latlng.LatLng(27.7011, 85.3532),
    "gausala": latlng.LatLng(27.7068, 85.3445),
    "chabhil": latlng.LatLng(27.7176, 85.3466),
    "gangabu": latlng.LatLng(27.7378, 85.3248),
    "samakhushi": latlng.LatLng(27.7347, 85.3143),
    "balaju": latlng.LatLng(27.7270, 85.3044),
    "banasthali": latlng.LatLng(27.7193, 85.2865),
    "swoyambhu": latlng.LatLng(27.7161, 85.2837),
    "satdobato": latlng.LatLng(27.6588, 85.3246),
    "gwarko": latlng.LatLng(27.6668, 85.3322),
    "balkumari": latlng.LatLng(27.6716, 85.3402),
  };

  @override
  void initState() {
    super.initState();
    _fabController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _fabScale = CurvedAnimation(
      parent: _fabController,
      curve: Curves.easeOutBack,
    );
    _fabController.forward();

    _locateMe();
    _checkLoginStatus();
    _loadRoutesOrStartSimulation();
  }

  Map<String, dynamic> _normalizeBus(dynamic b) {
    double? _toDouble(dynamic v) {
      if (v == null) return null;
      if (v is double) return v;
      if (v is int) return v.toDouble();
      if (v is String) return double.tryParse(v);
      return null;
    }

    try {
      if (b == null) return {};
      if (b is Map) {
        final lat = _toDouble(b['lat'] ?? b['latitude'] ?? b['y']);
        final lon = _toDouble(b['lon'] ?? b['longitude'] ?? b['x']);
        return {
          'bus_number':
              b['bus_number'] ?? b['bus_no'] ?? b['id'] ?? 'Unknown Bus',
          'lat': lat,
          'lon': lon,
          'eta_seconds_to_next': b['eta_seconds_to_next'] ?? b['eta'] ?? 0,
          'waiting_time_at_current': b['waiting_time_at_current'] ?? 0,
          'direction': b['direction'] ?? 'Unknown',
          'next_stop_name': b['next_stop_name'] ?? 'Unknown',
          ...b,
        };
      }
    } catch (e) {
      return {'bus_number': b.toString()};
    }
    return {'bus_number': b.toString()};
  }

  String _getPreviousStopName(Map<String, dynamic> bus) {
    final currentName = (bus['next_stop_name'] ?? 'unknown')
        .toString()
        .toLowerCase();
    final direction = bus['direction'] ?? 'Unknown';
    int currentIndex = stopsOrder.indexOf(currentName);
    if (currentIndex == -1) return "Unknown";

    if (direction == "Clockwise") {
      int prevIndex =
          (currentIndex - 1 + stopsOrder.length) % stopsOrder.length;
      return _capitalize(stopsOrder[prevIndex]);
    } else {
      int prevIndex = (currentIndex + 1) % stopsOrder.length;
      return _capitalize(stopsOrder[prevIndex]);
    }
  }

  String _capitalize(String s) => s[0].toUpperCase() + s.substring(1);

  Map<String, dynamic> _normalizeStop(dynamic stop) {
    if (stop is Map) {
      return {
        'name': stop['name'] ?? "Unknown Stop",
        'lat': stop['lat'],
        'lon': stop['lon'],
        'upcoming_buses': stop['upcoming_buses'] ?? [],
      };
    }
    return {'name': 'Unknown Stop', 'lat': null, 'lon': null};
  }

  @override
  void dispose() {
    _fabController.dispose();
    _updateTimer?.cancel();
    super.dispose();
  }

  Future<void> _checkLoginStatus() async {
    try {
      final user = await ApiService.me();
      if (user != null && mounted) {
        setState(() {
          _userName = user["name"]?.toString();
          _isAdmin = user["is_admin"] == true;
        });
      }
    } catch (e) {
      print("Login check error: $e");
    }
  }

  Future<void> _locateMe() async {
    final status = await Permission.location.request();
    if (status.isGranted) {
      try {
        final pos = await Geolocator.getCurrentPosition();
        if (mounted) {
          setState(() {
            _currentLocation = latlng.LatLng(pos.latitude, pos.longitude);
            _isLoadingLocation = false;
          });
          _mapController.move(_currentLocation, 16.0);
        }
      } catch (e) {
        if (mounted) setState(() => _isLoadingLocation = false);
      }
    } else {
      if (mounted) setState(() => _isLoadingLocation = false);
    }
  }

  Future<void> _loadRoutesOrStartSimulation() async {
    try {
      final url = AppConfig.routes;
      final resp = await http.get(Uri.parse(url));
      if (resp.statusCode == 200) {
        final routes = jsonDecode(resp.body) as List;
        if (routes.isNotEmpty) {
          setState(() => _availableRoutes = routes);
          final defaultRoute = routes.firstWhere(
            (r) => r['route_name'] == "Ring-Road-Ktm",
            orElse: () => routes.first,
          );
          await _selectRoute(defaultRoute);
          return;
        }
      }
    } catch (e) {
      print("Error loading routes: $e");
    }
    await _startServerRingRoadSimulation();
  }

  Future<void> _startServerRingRoadSimulation() async {
    try {
      final resp = await http.get(Uri.parse(AppConfig.ringRoadSimulation));
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        final simpleStops = (data['stops'] as List)
            .map((s) => _normalizeStop(s))
            .toList();

        setState(() {
          _selectedRoute = data['route'];
          _routePoints = (data['route']['coordinates'] as List)
              .map((c) => latlng.LatLng(c[1] as double, c[0] as double))
              .toList();
          _stops = simpleStops;
          _busesOnRoute = (data['buses'] as List)
              .map((b) => _normalizeBus(b))
              .toList();
          _simulatedBuses = [];
        });

        _updateTimer?.cancel();
        _updateTimer = Timer.periodic(const Duration(seconds: 1), (_) async {
          try {
            final update = await http.get(
              Uri.parse(AppConfig.ringRoadSimulation),
            );
            if (update.statusCode == 200 && mounted) {
              final newData = jsonDecode(update.body);
              final newStops = (newData['stops'] as List)
                  .map((s) => _normalizeStop(s))
                  .toList();
              setState(() {
                _busesOnRoute = (newData['buses'] as List)
                    .map((b) => _normalizeBus(b))
                    .toList();
                _stops = newStops;
              });
            }
          } catch (e) {
            print("Server simulation poll error: $e");
          }
        });
        return;
      }
    } catch (e) {
      print("Server simulation failed: $e");
    }
    _startRingRoadSimulation();
  }

  Future<void> _selectRoute(dynamic route) async {
    try {
      final resp = await http.get(Uri.parse(AppConfig.mapRoute(route['id'])));
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        _updateTimer?.cancel();
        setState(() {
          _simulatedBuses = [];
          _selectedRoute = data['route'];
          _stops = data['stops']; // Good (Accessing Map with String)
          _routePoints =
              (data['coordinates'] as List?)
                  ?.map((c) => latlng.LatLng(c[1] as double, c[0] as double))
                  .toList() ??
              [];
        });

        _updateTimer = Timer.periodic(const Duration(seconds: 1), (_) async {
          try {
            _startServerRingRoadSimulation();
            final updateResp = await http.get(
              Uri.parse(AppConfig.mapRoute(route['id'])),
            );
            if (updateResp.statusCode == 200 && mounted) {
              final updateData = jsonDecode(updateResp.body);
              final updatedStops = updateData["stops"];
              setState(() {
                _stops = updatedStops;
              });
            }
          } catch (e) {
            print("Error polling bus positions: $e");
          }
        });
      }
    } catch (e) {
      print("Error selecting route: $e");
      setState(() {
        _selectedRoute = null;
        _busesOnRoute = [];
        _stops = [];
        _routePoints = [];
        _simulatedBuses = [];
      });
      _startServerRingRoadSimulation();
    }
  }

  void _startRingRoadSimulation() {
    final routeCoordinates = <latlng.LatLng>[];
    for (final stopName in stopsOrder) {
      if (stopCenters.containsKey(stopName)) {
        routeCoordinates.add(stopCenters[stopName]!);
      }
    }
    if (routeCoordinates.isNotEmpty) {
      routeCoordinates.add(routeCoordinates[0]);
    }

    setState(() {
      _selectedRoute = null;
      _busesOnRoute = [];
      _stops = [];
      _routePoints = routeCoordinates;
    });

    _simulatedBuses = [
      SimulatedBus(
        id: "Bus 001",
        currentPosition: stopCenters["koteshwor"]!,
        currentStopIndex: 0,
        stopsOrder: stopsOrder,
        routePoints: routeCoordinates,
        progressIndex: 0,
      ),
      SimulatedBus(
        id: "Bus 002",
        currentPosition: stopCenters["chabhil"]!,
        currentStopIndex: 3,
        stopsOrder: stopsOrder,
        routePoints: routeCoordinates,
        progressIndex: 3,
      ),
      SimulatedBus(
        id: "Bus 003",
        currentPosition: stopCenters["balaju"]!,
        currentStopIndex: 7,
        stopsOrder: stopsOrder,
        routePoints: routeCoordinates,
        progressIndex: 7,
      ),
    ];

    _updateTimer?.cancel();
    _updateTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          for (var bus in _simulatedBuses) {
            bus.updatePosition(
              stopsOrder: stopsOrder,
              busStations: busStations,
              stopCenters: stopCenters,
            );
          }
        });
      }
    });
  }

  void _showRouteSelector() {
    if (_availableRoutes.isEmpty) return;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 5,
              margin: const EdgeInsets.only(top: 12),
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(4),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(20),
              child: Text(
                'Select Route',
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
              ),
            ),
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.5,
              child: ListView.builder(
                itemCount: _availableRoutes.length,
                itemBuilder: (_, i) {
                  final r = _availableRoutes[i];
                  final selected = _selectedRoute?['id'] == r['id'];
                  return ListTile(
                    leading: Icon(
                      Icons.route,
                      color: selected ? const Color(0xFF2E8B57) : Colors.grey,
                    ),
                    title: Text(r['route_name']),
                    trailing: selected
                        ? const Icon(Icons.check, color: Color(0xFF2E8B57))
                        : null,
                    onTap: () {
                      Navigator.pop(context);
                      _selectRoute(r);
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _showLogoutDialog() async {
    final shouldLogout = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(Icons.logout, color: Colors.red.shade700, size: 24),
            ),
            const SizedBox(width: 12),
            const Text(
              'Sign Out',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
            ),
          ],
        ),
        content: const Text(
          'Are you sure you want to sign out?',
          style: TextStyle(fontSize: 16),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red.shade600,
            ),
            child: const Text('Sign Out'),
          ),
        ],
      ),
    );
    if (shouldLogout == true) await _logout();
  }

  Future<void> _logout() async {
    await ApiService.logout();
    if (mounted) {
      setState(() {
        _userName = null;
        _isAdmin = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Signed out successfully'),
          backgroundColor: Colors.green.shade600,
        ),
      );
    }
  }

  void _centerOnLatLng(double lat, double lon) {
    _mapController.move(latlng.LatLng(lat, lon), 16.0);
  }

  void _showItemPopup(dynamic item, bool isBus) {
    setState(() {
      _selectedItem = item;
      _showPopup = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    final accent = const Color(0xFF2E8B57);
    final bool useServerData = _selectedRoute != null;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.blue.shade700.withOpacity(0.9),
                Colors.blue.shade900.withOpacity(0.9),
              ],
            ),
          ),
        ),
        leading: Builder(
          builder: (context) => IconButton(
            icon: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.menu, color: Colors.white, size: 24),
            ),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(
                Icons.directions_bus,
                color: Colors.white,
                size: 24,
              ),
            ),
            const SizedBox(width: 12),
            const Text(
              "Sadak-Sathi",
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.white,
                fontSize: 22,
              ),
            ),
          ],
        ),
        actions: [
          if (_availableRoutes.isNotEmpty)
            IconButton(
              icon: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.route, color: Colors.white),
              ),
              onPressed: _showRouteSelector,
              tooltip: "Select Route",
            ),
        ],
      ),
      drawer: _buildModernDrawer(_userName != null),
      body: Stack(
        children: [
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: _currentLocation,
              initialZoom: 13.0,
            ),
            children: [
              TileLayer(
                urlTemplate: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                tileProvider: CancellableNetworkTileProvider(),
              ),
              if (_routePoints.isNotEmpty)
                PolylineLayer(
                  polylines: [
                    Polyline(
                      points: _routePoints,
                      strokeWidth: 6,
                      color: accent.withOpacity(0.85),
                    ),
                  ],
                ),
              MarkerLayer(
                markers: [
                  Marker(
                    point: _currentLocation,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.blue.shade700.withOpacity(0.3),
                            blurRadius: 10,
                          ),
                        ],
                      ),
                      child: Icon(
                        Icons.my_location,
                        color: Colors.blue.shade700,
                        size: 32,
                      ),
                    ),
                  ),
                  ...(useServerData
                      ? _stops.map((stop) {
                          final double? lat = (stop['lat'] as num?)?.toDouble();
                          final double? lon = (stop['lon'] as num?)?.toDouble();
                          if (lat == null || lon == null) return null;
                          return Marker(
                            point: latlng.LatLng(lat, lon),
                            width: 80,
                            height: 80,
                            child: GestureDetector(
                              onTap: () {
                                _centerOnLatLng(lat, lon);
                                _showItemPopup(stop, false);
                              },
                              child: const Icon(
                                Icons.location_on,
                                color: Colors.orange,
                                size: 30,
                              ),
                            ),
                          );
                        }).whereType<Marker>()
                      : stopCenters.entries.map(
                          (entry) => Marker(
                            point: entry.value,
                            width: 80,
                            height: 80,
                            child: GestureDetector(
                              onTap: () {
                                _centerOnLatLng(
                                  entry.value.latitude,
                                  entry.value.longitude,
                                );
                                _showItemPopup({
                                  'name': entry.key,
                                  'lat': entry.value.latitude,
                                  'lon': entry.value.longitude,
                                }, false);
                              },
                              child: const Icon(
                                Icons.location_on,
                                color: Colors.orange,
                                size: 25,
                              ),
                            ),
                          ),
                        )),
                  ...(useServerData
                      ? _busesOnRoute.map((bus) {
                          final double? lat = (bus['lat'] as num?)?.toDouble();
                          final double? lon = (bus['lon'] as num?)?.toDouble();
                          if (lat == null || lon == null) return null;
                          return Marker(
                            point: latlng.LatLng(lat, lon),
                            width: 80,
                            height: 80,
                            child: GestureDetector(
                              onTap: () {
                                _centerOnLatLng(lat, lon);
                                _showItemPopup(bus, true);
                              },
                              child: Container(
                                decoration: const BoxDecoration(
                                  color: Colors.red,
                                  shape: BoxShape.circle,
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black38,
                                      blurRadius: 8,
                                      offset: Offset(0, 4),
                                    ),
                                  ],
                                ),
                                child: const Icon(
                                  Icons.directions_bus,
                                  color: Colors.white,
                                  size: 20,
                                ),
                              ),
                            ),
                          );
                        }).whereType<Marker>()
                      : _simulatedBuses.map(
                          (bus) => Marker(
                            point: bus.currentPosition,
                            width: 80,
                            height: 80,
                            child: GestureDetector(
                              onTap: () {
                                _centerOnLatLng(
                                  bus.currentPosition.latitude,
                                  bus.currentPosition.longitude,
                                );
                                _showItemPopup(bus, true);
                              },
                              child: Container(
                                decoration: const BoxDecoration(
                                  color: Colors.red,
                                  shape: BoxShape.circle,
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black38,
                                      blurRadius: 8,
                                      offset: Offset(0, 4),
                                    ),
                                  ],
                                ),
                                child: const Icon(
                                  Icons.directions_bus,
                                  color: Colors.white,
                                  size: 20,
                                ),
                              ),
                            ),
                          ),
                        )),
                ],
              ),
            ],
          ),

          // FABs
          Positioned(
            left: 16,
            bottom: 100,
            child: ScaleTransition(
              scale: _fabScale,
              child: Column(
                children: [
                  _buildModernFAB(
                    icon: Icons.search,
                    color: Colors.white,
                    iconColor: Colors.blue.shade700,
                    onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text("Search coming soon")),
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildModernExtendedFAB(
                    icon: Icons.emergency,
                    label: "SOS",
                    color: Colors.red.shade600,
                    onPressed: () => launchUrl(Uri.parse("tel:100")),
                  ),
                ],
              ),
            ),
          ),
          Positioned(
            right: 16,
            bottom: 100,
            child: ScaleTransition(
              scale: _fabScale,
              child: Column(
                children: [
                  _buildModernFAB(
                    icon: Icons.filter_list,
                    color: const Color(0xFFFFC400),
                    iconColor: Colors.black87,
                    onPressed: () => showModalBottomSheet(
                      context: context,
                      backgroundColor: Colors.transparent,
                      builder: (_) => const FilterPanel(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildModernFAB(
                    icon: _isLoadingLocation
                        ? Icons.hourglass_empty
                        : Icons.my_location,
                    color: Colors.white,
                    iconColor: Colors.blue.shade700,
                    onPressed: _locateMe,
                    isLoading: _isLoadingLocation,
                  ),
                ],
              ),
            ),
          ),

          // Expandable Bus List Panel
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeOutCubic,
              height: _isBusListExpanded ? 320 : 64,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(24),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 12,
                    offset: const Offset(0, -6),
                  ),
                ],
              ),
              child: Column(
                children: [
                  GestureDetector(
                    onTap: () => setState(
                      () => _isBusListExpanded = !_isBusListExpanded,
                    ),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      decoration: const BoxDecoration(
                        color: Color(0xFF2E8B57),
                        borderRadius: BorderRadius.vertical(
                          top: Radius.circular(24),
                        ),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            "Live Buses (${_busesOnRoute.length})",
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Icon(
                            _isBusListExpanded
                                ? Icons.keyboard_arrow_down_rounded
                                : Icons.keyboard_arrow_up_rounded,
                            color: Colors.white,
                            size: 28,
                          ),
                        ],
                      ),
                    ),
                  ),
                  Expanded(
                    child: _busesOnRoute.isEmpty
                        ? const Center(
                            child: Text(
                              "No active buses",
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.grey,
                              ),
                            ),
                          )
                        : ListView.builder(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 8,
                            ),
                            itemCount: _busesOnRoute.length,
                            itemBuilder: (context, i) {
                              final bus = _busesOnRoute[i];
                              final etaMin =
                                  ((bus['eta_seconds_to_next'] as num?) ?? 0) /
                                  60;
                              final lat = bus['lat'] as double?;
                              final lon = bus['lon'] as double?;

                              return Card(
                                elevation: 4,
                                margin: const EdgeInsets.symmetric(vertical: 6),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: InkWell(
                                  borderRadius: BorderRadius.circular(16),
                                  onTap: () {
                                    if (lat != null && lon != null) {
                                      _centerOnLatLng(lat, lon);
                                    }
                                  },
                                  child: Padding(
                                    padding: const EdgeInsets.all(12),
                                    child: Row(
                                      children: [
                                        Container(
                                          padding: const EdgeInsets.all(10),
                                          decoration: const BoxDecoration(
                                            color: Colors.red,
                                            shape: BoxShape.circle,
                                          ),
                                          child: const Icon(
                                            Icons.directions_bus,
                                            color: Colors.white,
                                            size: 24,
                                          ),
                                        ),
                                        const SizedBox(width: 12),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                bus['bus_number'] ??
                                                    'Bus ${i + 1}',
                                                style: const TextStyle(
                                                  fontWeight: FontWeight.bold,
                                                  fontSize: 16,
                                                ),
                                              ),
                                              const SizedBox(height: 4),
                                              Text(
                                                "From: ${_getPreviousStopName(bus)}",
                                                style: const TextStyle(
                                                  fontSize: 14,
                                                ),
                                              ),
                                              Text(
                                                "→ Next: ${bus['next_stop_name'] ?? 'Unknown'}",
                                                style: const TextStyle(
                                                  fontSize: 14,
                                                  fontWeight: FontWeight.w600,
                                                  color: Color(0xFF2E8B57),
                                                ),
                                              ),
                                              Text(
                                                "ETA: ${etaMin.toStringAsFixed(1)} min",
                                                style: const TextStyle(
                                                  fontSize: 14,
                                                ),
                                              ),
                                              if ((bus['waiting_time_at_current']
                                                          as num? ??
                                                      0) >
                                                  0)
                                                Text(
                                                  "Waiting: ${bus['waiting_time_at_current']} sec",
                                                  style: const TextStyle(
                                                    fontSize: 13,
                                                    color: Colors.orange,
                                                  ),
                                                ),
                                            ],
                                          ),
                                        ),
                                        Icon(
                                          Icons.chevron_right,
                                          color: Colors.grey.shade600,
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
                  ),
                ],
              ),
            ),
          ),

          // Popup
          if (_showPopup && _selectedItem != null)
            Positioned(
              bottom: _isBusListExpanded ? 340 : 80,
              left: 20,
              right: 20,
              child: Material(
                elevation: 12,
                borderRadius: BorderRadius.circular(20),
                color: Colors.white,
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            _selectedItem is SimulatedBus
                                ? _selectedItem.id
                                : (_selectedItem is Map
                                      ? (_selectedItem['bus_number'] ??
                                            _selectedItem['name'] ??
                                            'Item')
                                      : 'Item'),
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF2E8B57),
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.close, color: Colors.grey),
                            onPressed: () => setState(() => _showPopup = false),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      if (_selectedItem is Map &&
                          _selectedItem.containsKey('bus_number'))
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "Direction: ${_selectedItem['direction'] ?? 'Unknown'}",
                              style: const TextStyle(fontSize: 16),
                            ),
                            Text(
                              "From: ${_getPreviousStopName(_selectedItem)}",
                              style: const TextStyle(fontSize: 16),
                            ),
                            Text(
                              "→ Next: ${_selectedItem['next_stop_name'] ?? 'Unknown'}",
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              "ETA: ${((_selectedItem['eta_seconds_to_next'] as num? ?? 0) / 60).toStringAsFixed(1)} min",
                              style: const TextStyle(fontSize: 16),
                            ),
                            if ((_selectedItem['waiting_time_at_current']
                                        as num? ??
                                    0) >
                                0)
                              Text(
                                "Waiting: ${_selectedItem['waiting_time_at_current']} sec",
                                style: const TextStyle(fontSize: 16),
                              ),
                          ],
                        ),
                      if (_selectedItem is Map &&
                          _selectedItem.containsKey('upcoming_buses'))
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              "Upcoming Buses",
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF2E8B57),
                              ),
                            ),
                            const SizedBox(height: 8),
                            ...((_selectedItem['upcoming_buses']
                                        as List<dynamic>?) ??
                                    [])
                                .map(
                                  (bus) => Padding(
                                    padding: const EdgeInsets.symmetric(
                                      vertical: 4,
                                    ),
                                    child: Text(
                                      "• ${bus['bus_number']}: ~${bus['eta_minutes'].toStringAsFixed(1)} min",
                                      style: const TextStyle(fontSize: 16),
                                    ),
                                  ),
                                ),
                            if ((_selectedItem['upcoming_buses'] as List?)
                                    ?.isEmpty ??
                                true)
                              const Text(
                                "No upcoming buses",
                                style: TextStyle(
                                  fontSize: 16,
                                  fontStyle: FontStyle.italic,
                                ),
                              ),
                          ],
                        ),
                      if (_selectedItem is Map &&
                          !_selectedItem.containsKey('upcoming_buses') &&
                          !_selectedItem.containsKey('bus_number'))
                        Text(
                          "Stop: ${_selectedItem['name']}",
                          style: const TextStyle(fontSize: 16),
                        ),
                      if (_selectedItem is SimulatedBus)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "Current Stop: ${_selectedItem.currentStop}",
                              style: const TextStyle(fontSize: 16),
                            ),
                            Text(
                              "Next Stop: ${_selectedItem.nextStop}",
                              style: const TextStyle(fontSize: 16),
                            ),
                            Text(
                              "ETA: ${_selectedItem.etaMinutes.toStringAsFixed(1)} min",
                              style: const TextStyle(fontSize: 16),
                            ),
                          ],
                        ),
                      const SizedBox(height: 20),
                      Align(
                        alignment: Alignment.centerRight,
                        child: ElevatedButton.icon(
                          onPressed: () {
                            final pos = _selectedItem is SimulatedBus
                                ? _selectedItem.currentPosition
                                : latlng.LatLng(
                                    (_selectedItem['lat'] as num).toDouble(),
                                    (_selectedItem['lon'] as num).toDouble(),
                                  );
                            _mapController.move(pos, 16.0);
                          },
                          icon: const Icon(Icons.near_me),
                          label: const Text("Show on map"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF2E8B57),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildModernFAB({
    required IconData icon,
    required Color color,
    required Color iconColor,
    required VoidCallback onPressed,
    double size = 56,
    bool isLoading = false,
  }) {
    return Container(
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: color,
        shape: const CircleBorder(),
        child: InkWell(
          onTap: onPressed,
          customBorder: const CircleBorder(),
          child: Container(
            width: size,
            height: size,
            alignment: Alignment.center,
            child: isLoading
                ? const CircularProgressIndicator(
                    color: Colors.blue,
                    strokeWidth: 3,
                  )
                : Icon(icon, color: iconColor, size: size == 48 ? 22 : 26),
          ),
        ),
      ),
    );
  }

  Widget _buildModernExtendedFAB({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(30),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.4),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: color,
        borderRadius: BorderRadius.circular(30),
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(30),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: Colors.white, size: 22),
                const SizedBox(width: 8),
                Text(
                  label,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                    fontSize: 15,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildModernDrawer(bool isLoggedIn) {
    return Drawer(
      width: MediaQuery.of(context).size.width * 0.82,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.only(
          topRight: Radius.circular(32),
          bottomRight: Radius.circular(32),
        ),
      ),
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.blue.shade700, Colors.blue.shade900],
          ),
        ),
        child: Column(
          children: [
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  vertical: 40,
                  horizontal: 24,
                ),
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(6),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white30, width: 4),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.25),
                            blurRadius: 12,
                            offset: const Offset(0, 6),
                          ),
                        ],
                      ),
                      child: CircleAvatar(
                        radius: 50,
                        backgroundColor: Colors.white,
                        child: Icon(
                          isLoggedIn ? Icons.person : Icons.person_outline,
                          size: 64,
                          color: Colors.blue.shade700,
                        ),
                      ),
                    ),
                    const SizedBox(height: 22),
                    Text(
                      isLoggedIn ? "Hi, $_userName!" : "Welcome!",
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 26,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.18),
                        borderRadius: BorderRadius.circular(30),
                      ),
                      child: Text(
                        isLoggedIn
                            ? (_isAdmin ? "Administrator" : "User")
                            : "Sign in for more features",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const Divider(height: 1, color: Colors.white24),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.symmetric(
                  vertical: 16,
                  horizontal: 12,
                ),
                children: [
                  _drawerItem(
                    icon: Icons.favorite_border,
                    title: "Favorites",
                    onTap: () {},
                  ),
                  _drawerItem(
                    icon: Icons.history,
                    title: "Recent Searches",
                    onTap: () {},
                  ),
                  _drawerItem(
                    icon: Icons.settings_outlined,
                    title: "Settings",
                    onTap: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const SettingsScreen()),
                    ),
                  ),
                  _drawerItem(
                    icon: Icons.help_outline,
                    title: "Help & Support",
                    onTap: () {},
                  ),
                  if (_isAdmin)
                    _drawerItem(
                      icon: Icons.admin_panel_settings,
                      title: "Admin Panel",
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const AdminFeaturePage(),
                        ),
                      ),
                      isHighlighted: true,
                    ),
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 12),
                    child: Divider(color: Colors.white24),
                  ),
                  if (isLoggedIn)
                    _drawerItem(
                      icon: Icons.logout,
                      title: "Sign Out",
                      onTap: _showLogoutDialog,
                      isDanger: true,
                    )
                  else
                    _drawerItem(
                      icon: Icons.login,
                      title: "Sign In / Register",
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const AuthScreen()),
                      ),
                    ),
                ],
              ),
            ),
            SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  "Sadak-Sathi • v1.0",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.55),
                    fontSize: 13,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _drawerItem({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
    bool isHighlighted = false,
    bool isDanger = false,
  }) {
    final itemColor = isDanger
        ? Colors.red.shade300
        : isHighlighted
        ? Colors.amber.shade300
        : Colors.white;
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        color: Colors.white.withOpacity(0.05),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () {
            Navigator.pop(context);
            onTap();
          },
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            child: Row(
              children: [
                Icon(icon, color: itemColor, size: 24),
                const SizedBox(width: 20),
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(
                      color: itemColor,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                Icon(
                  Icons.chevron_right,
                  color: itemColor.withOpacity(0.5),
                  size: 20,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class SimulatedBus {
  final String id;
  final List<latlng.LatLng> routePoints;
  latlng.LatLng currentPosition;
  String currentStop = "koteshwor";
  String nextStop = "airport";
  double etaMinutes = 0;
  int currentStopIndex = 0;
  int progressIndex = 0;

  SimulatedBus({
    required this.id,
    required this.currentPosition,
    required this.currentStopIndex,
    required List<String> stopsOrder,
    required this.routePoints,
    this.progressIndex = 0,
  }) {
    if (currentStopIndex < stopsOrder.length) {
      currentStop = stopsOrder[currentStopIndex];
      nextStop = currentStopIndex < stopsOrder.length - 1
          ? stopsOrder[currentStopIndex + 1]
          : stopsOrder[0];
    }
  }

  Future<void> updatePosition({
    required List<String> stopsOrder,
    required Map<String, List<double>> busStations,
    required Map<String, latlng.LatLng> stopCenters,
  }) async {
    if (routePoints.isNotEmpty) {
      progressIndex = (progressIndex + 1) % routePoints.length;
      currentPosition = routePoints[progressIndex];
      currentStop = "Moving";

      for (final entry in busStations.entries) {
        final box = entry.value;
        if (currentPosition.latitude >= box[0] &&
            currentPosition.latitude <= box[2] &&
            currentPosition.longitude >= box[1] &&
            currentPosition.longitude <= box[3]) {
          currentStop = entry.key;
          currentStopIndex = stopsOrder.indexOf(currentStop);
          break;
        }
      }

      nextStop =
          currentStopIndex >= 0 && currentStopIndex < stopsOrder.length - 1
          ? stopsOrder[currentStopIndex + 1]
          : (stopsOrder.isNotEmpty ? stopsOrder[0] : "Unknown");
    }

    try {
      final response = await http.post(
        Uri.parse(AppConfig.predictEta),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "from_stop": currentStopIndex >= 0 ? currentStopIndex : 0,
          "to_stop": (currentStopIndex + 1) % stopsOrder.length,
          "dist": 5000.0,
          "dir_ref": 1,
          "rush": DateTime.now().hour >= 7 && DateTime.now().hour <= 9 ? 1 : 0,
        }),
      );
      if (response.statusCode == 200) {
        etaMinutes = jsonDecode(response.body)["eta_seconds"] / 60;
      }
    } catch (e) {
      etaMinutes = 8.0;
    }
  }
}
