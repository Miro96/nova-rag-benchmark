# Complex Query Ground Truth

This document defines the ground truth for 15 complex code-intelligence queries
(5 per repo: Flask, FastAPI, Express). Each query targets a gap category not
covered by the existing 90 queries.

All `expected_files` paths are verified to exist in the cloned repos at
`~/.cache/rag-bench/repos/{flask,fastapi,express}/`.

## Gap Category Coverage

| Type              | Count | Repos            |
|-------------------|-------|------------------|
| multi_hop         | 3     | Flask, FastAPI, Express |
| cross_package     | 3     | Flask, FastAPI, Express |
| architecture      | 3     | Flask, FastAPI, Express |
| impact            | 3     | Flask, FastAPI, Express |
| dead_code         | 1     | Flask            |
| conditional_path  | 1     | FastAPI          |
| test_traceability | 1     | Express          |

---

## Flask Queries

### flask_031 — multi_hop
**Query:** Trace the full lifecycle of a URL route in Flask: from the @app.route decorator through URL rule registration, request matching, and final dispatch to the view function. Which modules and functions form each hop?

**Expected files:**
1. `src/flask/app.py` — Flask.route, add_url_rule, full_dispatch_request, dispatch_request
2. `src/flask/sansio/scaffold.py` — Scaffold.route, _endpoint_from_view_func
3. `src/flask/sansio/app.py` — App.add_url_rule, App.url_map
4. `src/flask/ctx.py` — RequestContext (push/pop around dispatch)
5. `src/flask/globals.py` — _cv_request, request proxy

**Expected symbols:** `route`, `add_url_rule`, `full_dispatch_request`, `dispatch_request`, `Scaffold`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Flask repo.

---

### flask_032 — cross_package
**Query:** How does Flask's sansio (IO-free) core package bridge to the WSGI-aware layer? Trace the cross-package dependencies from flask.sansio to flask.app, and identify the boundary modules that separate protocol-agnostic logic from HTTP/WSGI concerns.

**Expected files:**
1. `src/flask/sansio/app.py` — App base class (no WSGI)
2. `src/flask/sansio/scaffold.py` — Scaffold base class
3. `src/flask/sansio/blueprints.py` — Blueprint sansio logic
4. `src/flask/app.py` — Flask (WSGI-aware subclass of App)
5. `src/flask/wrappers.py` — Request/Response (WSGI wrappers)

**Expected symbols:** `App`, `Flask`, `Scaffold`, `Request`, `Response`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Flask repo.

---

### flask_033 — architecture
**Query:** Describe Flask's layered architecture: how do the sansio core, the WSGI application layer, the context stack (app/request contexts), the signals/event system, and the extension loading mechanism compose into a cohesive web framework?

**Expected files:**
1. `src/flask/sansio/app.py` — Sansio application core
2. `src/flask/app.py` — WSGI application layer
3. `src/flask/ctx.py` — Context stack (AppContext, RequestContext)
4. `src/flask/signals.py` — Blinker signal definitions
5. `src/flask/globals.py` — Thread/context-local proxies

**Expected symbols:** `App`, `Flask`, `AppContext`, `request_started`, `current_app`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Flask repo.

---

### flask_034 — impact
**Query:** If the Scaffold.before_request hook registration and trigger mechanism is changed, perform an impact analysis: which modules, classes, and downstream systems (Blueprints, error handling, testing) would be affected?

**Expected files:**
1. `src/flask/sansio/scaffold.py` — Scaffold.before_request decorator, try_trigger_before_request_functions
2. `src/flask/app.py` — Flask.try_trigger_before_request_functions, full_dispatch_request
3. `src/flask/sansio/app.py` — App base class (inherits from Scaffold)
4. `src/flask/blueprints.py` — Blueprint (also inherits from Scaffold)
5. `src/flask/sansio/blueprints.py` — Blueprint sansio layer

**Expected symbols:** `Scaffold`, `before_request`, `try_trigger_before_request_functions`, `Blueprint`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Flask repo.

---

### flask_035 — dead_code
**Query:** Analyze Flask's JSON provider system for potentially unused or rarely-triggered code paths. The JSONProvider and DefaultJSONProvider classes have methods for serialization customization — which methods or code paths are defined but rarely exercised in typical Flask request flows?

**Expected files:**
1. `src/flask/json/provider.py` — DefaultJSONProvider, JSONProvider base
2. `src/flask/json/__init__.py` — jsonify, dumps, loads convenience functions
3. `src/flask/json/tag.py` — TaggedJSONSerializer (for non-standard types)
4. `src/flask/app.py` — Flask.json_provider_class, json assignment
5. `src/flask/wrappers.py` — Request.get_json, Response JSON handling

**Expected symbols:** `DefaultJSONProvider`, `JSONProvider`, `TaggedJSONSerializer`, `jsonify`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Flask repo.

---

## FastAPI Queries

### fastapi_031 — multi_hop
**Query:** Trace how a path parameter declared in a FastAPI route handler flows through the dependency injection system: from parameter declaration through Dependant model construction, dependency solving, request validation, and final injection into the endpoint function. Identify each hop.

**Expected files:**
1. `fastapi/routing.py` — APIRoute, get_request_handler, run_endpoint_function
2. `fastapi/dependencies/utils.py` — solve_dependencies, get_dependant, get_flat_dependant
3. `fastapi/dependencies/models.py` — Dependant, SecurityRequirement
4. `fastapi/params.py` — Path, Query, Depends parameter classes
5. `fastapi/utils.py` — get_path_param_names, create_model_field

**Expected symbols:** `get_dependant`, `solve_dependencies`, `get_request_handler`, `Dependant`, `get_path_param_names`

**Verified files exist:** ✅ All 5 paths confirmed in cloned FastAPI repo.

---

### fastapi_032 — cross_package
**Query:** How does FastAPI bridge between Starlette's ASGI routing/request handling and its own dependency injection, validation, and OpenAPI layers? Trace the boundary where Starlette's Request is transformed into FastAPI's typed parameter injection.

**Expected files:**
1. `fastapi/routing.py` — APIRoute (extends starlette.routing.Route)
2. `fastapi/dependencies/utils.py` — solve_dependencies (bridges Starlette connection to FastAPI params)
3. `fastapi/applications.py` — FastAPI (extends starlette.applications.Starlette)
4. `fastapi/openapi/utils.py` — get_openapi (cross-package: routing → openapi)
5. `fastapi/dependencies/models.py` — Dependant model (shared between routing and deps)

**Expected symbols:** `APIRoute`, `get_request_handler`, `solve_dependencies`, `get_openapi`, `FastAPI`

**Verified files exist:** ✅ All 5 paths confirmed in cloned FastAPI repo.

---

### fastapi_033 — architecture
**Query:** Describe the layered architecture of FastAPI: how do Starlette's ASGI core, FastAPI's dependency injection system, Pydantic's data validation, and the OpenAPI documentation generation compose into a cohesive API framework?

**Expected files:**
1. `fastapi/applications.py` — FastAPI application class (entry point)
2. `fastapi/routing.py` — APIRoute, routing layer
3. `fastapi/dependencies/utils.py` — Dependency injection engine
4. `fastapi/openapi/utils.py` — OpenAPI schema generation
5. `fastapi/params.py` — Parameter type hierarchy (Path, Query, Body, Depends)

**Expected symbols:** `FastAPI`, `APIRoute`, `solve_dependencies`, `get_openapi`, `Depends`

**Verified files exist:** ✅ All 5 paths confirmed in cloned FastAPI repo.

---

### fastapi_034 — impact
**Query:** If the OpenAPI schema generation function (get_openapi) is modified, perform an impact analysis: which downstream systems (Swagger UI, ReDoc, client code generation, FastAPI's own routing metadata) and user-facing features would be affected?

**Expected files:**
1. `fastapi/openapi/utils.py` — get_openapi (core schema generation)
2. `fastapi/openapi/docs.py` — get_swagger_ui_html, get_redoc_html (consumes OpenAPI)
3. `fastapi/applications.py` — FastAPI.openapi (cached schema access)
4. `fastapi/routing.py` — APIRoute (contributes path operation metadata)
5. `fastapi/openapi/models.py` — OpenAPI data models

**Expected symbols:** `get_openapi`, `get_swagger_ui_html`, `FastAPI`, `APIRoute`

**Verified files exist:** ✅ All 5 paths confirmed in cloned FastAPI repo.

---

### fastapi_035 — conditional_path
**Query:** How does FastAPI conditionally handle response serialization? Trace the branching logic that determines when jsonable_encoder is called vs direct Pydantic model_dump, and what factors (response_model, response_class, return type annotation) determine which code path is taken.

**Expected files:**
1. `fastapi/routing.py` — serialize_response, conditional branching for response handling
2. `fastapi/encoders.py` — jsonable_encoder (custom serialization path)
3. `fastapi/dependencies/utils.py` — get_typed_return_annotation, response model extraction
4. `fastapi/_compat.py` — _model_dump, Pydantic v1/v2 compatibility branches
5. `fastapi/responses.py` — JSONResponse, custom response classes

**Expected symbols:** `serialize_response`, `jsonable_encoder`, `_model_dump`, `JSONResponse`

**Verified files exist:** ✅ All 5 paths confirmed in cloned FastAPI repo.

---

## Express Queries

### express_031 — multi_hop
**Query:** Trace how req.ip is computed in Express 5: from the trust proxy configuration setting through compileTrust validation, to the X-Forwarded-For header parsing and final IP extraction. Identify each hop in the computation chain.

**Expected files:**
1. `lib/request.js` — req.ip getter, trust proxy evaluation
2. `lib/utils.js` — compileTrust function
3. `lib/application.js` — app.set('trust proxy'), trust proxy configuration
4. `lib/express.js` — createApplication, app initialization
5. `index.js` — package entry point

**Expected symbols:** `req.ip`, `compileTrust`, `trust proxy`, `createApplication`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Express repo.

---

### express_032 — cross_package
**Query:** How does Express 5 delegate routing to the external 'router' npm package? Trace the boundary where application.js and express.js integrate the external Router, and how the router's handle method connects back to Express's request/response objects.

**Expected files:**
1. `lib/express.js` — require('router'), Router integration, createApplication
2. `lib/application.js` — app.handle, app.lazyrouter, Router usage
3. `lib/response.js` — res methods consumed by router middleware chain
4. `lib/request.js` — req properties consumed by router matching
5. `index.js` — package exports (exposes Router constructor)

**Expected symbols:** `Router`, `createApplication`, `handle`, `lazyrouter`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Express repo.

---

### express_033 — architecture
**Query:** Describe the overall architecture of Express 5: how does the application factory (createApplication), the request/response prototype chain, the view rendering system, and the delegated router compose into the framework?

**Expected files:**
1. `lib/application.js` — Application prototype (settings, middleware, listen)
2. `lib/express.js` — createApplication, mixin composition, exports
3. `lib/request.js` — req prototype (IP, headers, content negotiation)
4. `lib/response.js` — res prototype (send, json, redirect, render)
5. `lib/view.js` — View class (template rendering engine)

**Expected symbols:** `createApplication`, `app`, `req`, `res`, `View`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Express repo.

---

### express_034 — impact
**Query:** If res.send is modified, perform an impact analysis: which other response methods (json, jsonp, sendFile, sendStatus, redirect, render) and application-level behaviors would be affected by the change?

**Expected files:**
1. `lib/response.js` — res.send, res.json, res.jsonp, res.sendFile, res.sendStatus
2. `lib/application.js` — app.render (delegates to res.render), implicit send
3. `lib/express.js` — response prototype wiring
4. `lib/request.js` — req.accepts (used by res.format which calls send)
5. `lib/utils.js` — contentDisposition, content-type helpers used by send/sendFile

**Expected symbols:** `res.send`, `res.json`, `res.jsonp`, `res.sendFile`, `res.sendStatus`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Express repo.

---

### express_035 — test_traceability
**Query:** Which test files and test suites validate Express's cookie setting and clearing behavior? Map the res.cookie and res.clearCookie implementations to their corresponding test coverage files.

**Expected files:**
1. `lib/response.js` — res.cookie, res.clearCookie implementation
2. `test/res.cookie.js` — Cookie setting test suite
3. `test/res.clearCookie.js` — Cookie clearing test suite
4. `lib/application.js` — app-level settings that affect cookies
5. `lib/request.js` — req.signedCookies, req.cookies parsing

**Expected symbols:** `res.cookie`, `res.clearCookie`, `cookie`

**Verified files exist:** ✅ All 5 paths confirmed in cloned Express repo.
