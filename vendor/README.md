# vendor

`lexapi-client-4f3761b.tgz` is `@lexapi/client` built from
[Lex-API/lexapi-typescript](https://github.com/Lex-API/lexapi-typescript) at commit
`4f3761b52f965eaee5bc01390f4075891a679cfd`. That version is not on npm yet, and installing the
package from git does not build it, so `package.json` installs this build. Its `package.json` still
says 0.4.0.

To rebuild it and compare:

```bash
git clone https://github.com/Lex-API/lexapi-typescript && cd lexapi-typescript
git checkout 4f3761b52f965eaee5bc01390f4075891a679cfd
npm ci && npm run build && npm pack
shasum -a 256 lexapi-client-0.4.0.tgz
# 1a2cb6ebc7c93fe7550aa3d06d1391f6dfc473a64445d9a05ce91ba384d50778 (npm 10.9, Node 22)
```

Once the release is on npm, `package.json` depends on `@lexapi/client@^0.5.0` and this folder is
removed.
