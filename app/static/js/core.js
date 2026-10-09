// core.js — phần dùng chung của annotator: canvas, khung nhìn, nạp ảnh,
// điều hướng, thanh trạng thái và sổ đăng ký các phần giao diện.
// Các file JS là script thường, nạp theo thứ tự trong index.html và dùng chung
// phạm vi toàn cục. File chỉ ĐỊNH NGHĨA; gắn sự kiện và khởi động nằm ở main.js.
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),wrap=document.getElementById('wrap');
const $=id=>document.getElementById(id);
function enc(t){return t.split('/').map(encodeURIComponent).join('/');}
function natcmp(a,b){return String(a).localeCompare(String(b),undefined,{numeric:true});}
let imgs=[],labeledSet=new Set(),current=null,img=new Image(),curFolder='',pathFilter='';
let baseFit=1,zoom=1,scale=1,ox=0,oy=0;
let selToken=0,ready=false;
// --- sổ đăng ký -------------------------------------------------------------
// Mỗi phần giao diện (HSV/Otsu, sửa tán, dự đoán…) đăng ký một đối tượng có
// các móc tuỳ chọn; core gọi chúng theo thứ tự đăng ký. Thứ tự đó cũng là thứ
// tự VẼ: lớp pixel hoa, rồi tán, rồi dự đoán đè lên trên cùng.
//   beforeImage()        trước khi rời ảnh đang mở (lưu nốt phần đang sửa)
//   resetImage(t)        dọn trạng thái của ảnh cũ, đồng bộ
//   loadImage(t, tok)    nạp dữ liệu của ảnh mới (async); tok đổi là bỏ
//   imageShown()         ảnh đã tải xong và khung nhìn đã đặt lại
//   onImage(t, tok)      dữ liệu đã sẵn sàng (ready = true)
//   draw(ctx)            vẽ lớp của mình lên canvas
const WS=[];
function wsRegister(w){WS.push(w);return w;}
function wsEach(name,...args){for(const w of WS)if(w[name])w[name](...args);}
// Thống kê bám theo thư mục đang chọn: đang gán dở một nhánh thì cần biết còn
// bao nhiêu ảnh của NHÁNH ĐÓ, chứ tổng toàn bộ không giúp gì.
function renderStats(){
 const list=pathFilter?imgs.filter(inScope):imgs;
 const total=list.length,lab=list.filter(p=>labeledSet.has(p)).length;
 $('stTotal').textContent=total;$('stLab').textContent=lab;
 $('stUnlab').textContent=Math.max(0,total-lab);$('stCls').textContent=classes.length;
 const sub=pathFilter?(' · '+pathFilter):' · tất cả';
 document.querySelectorAll('#leftcol .stat .slbl')[0].textContent='Total Images'+sub;}
async function loadLabeled(){try{const d=await(await fetch('/api/labeled?t='+Date.now(),{cache:'no-store'})).json();labeledSet=new Set(d.labeled||[]);emptySet=new Set(d.empty||[]);}catch(e){labeledSet=new Set();emptySet=new Set();}renderStats();}
// Cây thư mục: mỗi nhánh mang sẵn số ảnh và số đã gán, để biết còn phải làm
// bao nhiêu mà không cần bấm vào từng thư mục.
function buildDirs(paths){const root={dirs:{},n:0,done:0};
 paths.forEach(p=>{const parts=p.split('/');let node=root;
  for(let i=0;i<parts.length-1;i++){node.dirs[parts[i]]=node.dirs[parts[i]]||{dirs:{},n:0,done:0};
   node=node.dirs[parts[i]];node.n++;if(labeledSet.has(p))node.done++;}
  root.n++;if(labeledSet.has(p))root.done++;});
 return root;}
let collapsed=new Set();     // giữ qua các lần vẽ lại, nếu không mỗi lần bấm là bung hết
function firstImgIn(pre){return imgs.find(p=>pre===''||p===pre||p.indexOf(pre+'/')===0);}
function pickDir(full){pathFilter=full;renderPTree();renderThumbs();renderStats();
 const f=firstImgIn(full);if(f&&f!==current)selectImg(f);}
function treeRow(label,full,node){
 const row=document.createElement('div');
 const kidNames=Object.keys(node.dirs);
 row.className='dir'+(pathFilter===full?' on':'')+(node.n&&node.done===node.n?' done':'');
 const tw=document.createElement('span');tw.className='tw';
 if(kidNames.length)tw.textContent=collapsed.has(full)?'▸':'▾';   // lá thì không có mũi
 row.appendChild(tw);
 const nm=document.createElement('span');nm.className='dn';nm.textContent='📁 '+label;row.appendChild(nm);
 const c=document.createElement('span');c.className='dc';
 c.textContent=node.done+'/'+node.n;row.title=label+' — đã gán '+node.done+'/'+node.n+' ảnh';
 row.appendChild(c);
 row.onclick=e=>{e.stopPropagation();pickDir(full);};
 if(kidNames.length)tw.onclick=e=>{e.stopPropagation();
  if(collapsed.has(full))collapsed.delete(full);else collapsed.add(full);renderPTree();};
 return row;}
function progBar(node){const b=document.createElement('div');b.className='bar';
 const i=document.createElement('i');i.style.width=(node.n?100*node.done/node.n:0)+'%';
 b.appendChild(i);return b;}
function renderPTree(){const el=$('ptree');el.innerHTML='';
 const root=buildDirs(imgs);
 el.appendChild(treeRow('Tất cả','',root));
 el.appendChild(progBar(root));
 if(collapsed.has(''))return;
 const wrap2=document.createElement('div');wrap2.className='kids';
 (function walk(node,prefix,cont){Object.keys(node.dirs).sort(natcmp).forEach(name=>{
   const full=prefix?prefix+'/'+name:name,kid=node.dirs[name];
   cont.appendChild(treeRow(name,full,kid));
   if(collapsed.has(full))return;
   const kids=document.createElement('div');kids.className='kids';
   walk(kid,full,kids);cont.appendChild(kids);});})(root,'',wrap2);
 el.appendChild(wrap2);}
function isDone(p){return labeledSet.has(p);}
function inScope(p){return pathFilter===''||p===pathFilter||p.indexOf(pathFilter+'/')===0;}
function renderThumbs(){const box=$('thumbs');box.innerHTML='';
 const q=($('search').value||'').toLowerCase();
 const list=imgs.filter(p=>inScope(p)&&p.toLowerCase().indexOf(q)>=0);
 // Hiện số ĐANG LỌC, không phải tổng — bản cũ luôn ghi tổng nên chọn thư mục
 // 67 ảnh vẫn thấy 118.
 $('nimg').textContent=(list.length===imgs.length)?imgs.length:(list.length+' / '+imgs.length);
 list.forEach(p=>{const row=document.createElement('div');row.className='thumb'+(p===current?' active':'')+(isDone(p)?' done':'')+(emptySet.has(p)?' empty':'');row.dataset.n=p;
  const im=document.createElement('img');im.loading='lazy';im.src='/api/thumb/'+enc(p);
  const nm=document.createElement('div');nm.className='tname';nm.textContent=p.split('/').pop();nm.title=p;
  row.appendChild(im);row.appendChild(nm);row.onclick=()=>selectImg(p);box.appendChild(row);});}
function updateThumbActive(){document.querySelectorAll('#thumbs .thumb').forEach(t=>{t.classList.toggle('active',t.dataset.n===current);t.classList.toggle('done',isDone(t.dataset.n));t.classList.toggle('empty',emptySet.has(t.dataset.n));});}
async function loadImgs(){imgs=await(await fetch('/api/images',{cache:'no-store'})).json();
 await loadLabeled();renderPTree();renderThumbs();renderStats();
 if(imgs.length)selectImg(imgs[0]);else{inst=[];current=null;draw();renderRegions();}}
// Đổi ảnh. `tok` chốt lần đổi: hai lần đổi chồng nhau thì lần về sau không
// được ghi đè bằng kết quả của ảnh trước (mọi phần nạp dữ liệu đều hỏi lại).
async function selectImg(t){wsEach('beforeImage');
 const tok=++selToken;
 current=t;ready=false;wsEach('resetImage',t);updateThumbActive();
 img=new Image();img.onload=()=>{if(tok===selToken){sizeCanvas();resetView();wsEach('imageShown');}};
 img.src='/api/image/'+enc(t)+'?t='+Date.now();
 for(const w of WS){if(!w.loadImage)continue;await w.loadImage(t,tok);if(tok!==selToken)return;}
 ready=true;wsEach('onImage',t,tok);draw();}
// Chỉ báo lưu. Bản cũ gọi fetch rồi bỏ qua kết quả, nên server có báo lỗi cũng
// không ai thấy — kể cả khi nó không đọc được ảnh và không tính được mật độ hoa.
function svst(cls,txt,title){const el=$('svst');if(!el)return;
 el.className=cls||'';el.textContent=txt||'';el.title=title||txt||'';}
function hhmm(){const d=new Date(),p=n=>('0'+n).slice(-2);
 return p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());}
function sizeCanvas(){cv.width=wrap.clientWidth;cv.height=wrap.clientHeight;}
function resetView(){if(!img.width)return;baseFit=Math.min(cv.width/img.width,cv.height/img.height,1);zoom=1;scale=baseFit;ox=(cv.width-img.width*scale)/2;oy=(cv.height-img.height*scale)/2;setZlbl();draw();}
function zoomAt(cx,cy,f){if(!img.width)return;const ix=(cx-ox)/scale,iy=(cy-oy)/scale;zoom=Math.min(12,Math.max(0.1,zoom*f));scale=baseFit*zoom;ox=cx-ix*scale;oy=cy-iy*scale;setZlbl();draw();}
function setZlbl(){$('zlbl').textContent=Math.round(zoom*100)+'%';}
function toImg(e){const r=cv.getBoundingClientRect();const kx=r.width?cv.width/r.width:1,ky=r.height?cv.height/r.height:1;return[((e.clientX-r.left)*kx-ox)/scale,((e.clientY-r.top)*ky-oy)/scale];}
// Kẹp toạ độ về trong khung ảnh: kéo/vẽ tràn ra ngoài thì điểm dán vào mép
// chứ không nằm ngoài ảnh. Biên [0,w]/[0,h] khớp đúng _clip_poly bên server.
function cX(x){return img.width?Math.min(Math.max(x,0),img.width):x;}
function cY(y){return img.height?Math.min(Math.max(y,0),img.height):y;}
function sx(x){return x*scale+ox;}
function sy(y){return y*scale+oy;}
function hexA(h,a){const n=parseInt(h.slice(1),16);return'rgba('+(n>>16&255)+','+(n>>8&255)+','+(n&255)+','+a+')';}
function centroid(p){let x=0,y=0,n=p.length/2;for(let i=0;i<p.length;i+=2){x+=p[i];y+=p[i+1];}return[x/n,y/n];}
function polyPath(p){ctx.beginPath();
 for(let k=0;k<p.length;k+=2){k===0?ctx.moveTo(sx(p[k]),sy(p[k+1])):ctx.lineTo(sx(p[k]),sy(p[k+1]));}
 ctx.closePath();}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);
 if(img.complete&&img.width)ctx.drawImage(img,ox,oy,img.width*scale,img.height*scale);
 for(const w of WS)if(w.draw)w.draw(ctx);}
function pip(p,x,y){let c=false;for(let i=0,j=p.length-2;i<p.length;j=i,i+=2){const xi=p[i],yi=p[i+1],xj=p[j],yj=p[j+1];if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))c=!c;}return c;}
// Phím tắt chỉ dành cho canvas. Đang gõ trong ô tìm kiếm / ô số Otsu / dropdown
// lớp thì trả phím về cho ô đó: trước đây gõ "0" là reset zoom, "-" là zoom
// out, Delete là xoá vùng đang chọn (kể cả vùng đang ẩn, không thấy nó mất).
function typing(e){const t=e.target,tag=t&&t.tagName;
 return tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT'||(t&&t.isContentEditable);}
