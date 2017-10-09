
var curY= function(){
	if(typeof(window.pageYoffset) == 'number')
		return window.pageYoffset;
	else if(typeof(document.documentElement.scrollTop) == 'number')
		return document.documentElement.scrollTop;
	else 
		return 0;
};

var navi_lock = function()
{
	navi = document.getElementById('navi');
	logo = document.getElementById('logo');

	window.onscroll = function()
	{
		if(curY() <= navi.offsetTop){
			if(navi.style.position == 'fixed'){
				logo.style.display = 'block';
				navi.style.position = 'static';
				scrollTo(0, 200);
			}
		}
		else{
			if(navi.style.position != 'fixed'){
				logo.style.display = 'none';
				navi.style.position = 'fixed';
			}
		}
	};
};
