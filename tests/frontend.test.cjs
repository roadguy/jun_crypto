const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {textContent:'', style:{}, classList:{remove(){},add(){},toggle(){}},replaceChildren(){},value: id === 'timeframe' ? '4h' : 'BTC'});
  return elements.get(id);
};
const context = {
  document:{getElementById:element, querySelectorAll:()=>[]},
  location:{protocol:'http:'}, AbortController, setTimeout, clearTimeout, console,
  fetch:async()=>({ok:false,status:422,json:async()=>({detail:[{msg:'invalid token'}]})}),
};
vm.createContext(context);
let source = fs.readFileSync(require('node:path').join(__dirname,'../static/app.js'),'utf8');
source = source.slice(0,source.indexOf('\ndocument.querySelectorAll("[data-action]")'));
vm.runInContext(source,context);
assert.equal(vm.runInContext('fmt.pct(null)',context),'—');
assert.equal(vm.runInContext('fmt.price(undefined)',context),'—');
assert.equal(vm.runInContext('fmt.pct(0)',context),'0.00%');
vm.runInContext('renderRange({})',context);
assert.equal(element('q10Price').textContent,'— USDT');
assert.equal(element('coverage').textContent,'—');
(async()=>{
  await assert.rejects(vm.runInContext('requestJson("/api/jobs")',context), /invalid token/);
  context.location.protocol='file:';
  await assert.rejects(vm.runInContext('requestJson("/api/meta")',context), /HTML 파일을 직접/);
  console.log('Frontend formatting, missing range, API error, file URL checks passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
