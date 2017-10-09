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
				banner.style.display = 'block';
				navi.style.position = 'static';
				scrollTo(0, navi.offsetTop);
			}
		}
		else{
			if(navi.style.position != 'fixed'){
				banner.style.display = 'none';
				navi.style.position = 'fixed';
			}
		}
	};
};

function scrollToItem(item) {
    var diff=(item.offsetTop - window.scrollY)/8
    if (Math.abs(diff)>1) {
        window.scrollTo(0, (window.scrollY+diff))
        clearTimeout(window._TO)
        window._TO=setTimeout(scrollToItem, 30, item)
    } else {
        window.scrollTo(0, item.offsetTop)
    }
}

var navi_click = function()
{
	n_top = document.getElementById('navi_top');
	n_2016 = document.getElementById('navi_2016');
	n_2015 = document.getElementById('navi_2015');
	n_2014 = document.getElementById('navi_2014');
	n_2013 = document.getElementById('navi_2013');
	n_others = document.getElementById('navi_others');
	
	n_top.onclick = function()
	{
		section = document.getElementById('navi');
		//scrollTo(0, section.offsetTop);
		scrollToItem(section);
	}
	
	n_2016.onclick = function()
	{
		section = document.getElementById('2016');
		//scrollTo(0, section.offsetTop + 150);
		scrollToItem(section);
	}
	
	n_2015.onclick = function()
	{
		section = document.getElementById('2015');
		//scrollTo(0, section.offsetTop + 150);
		scrollToItem(section);
	}
	
	n_2014.onclick = function()
	{
		section = document.getElementById('2014');
		//scrollTo(0, section.offsetTop + 150);
		scrollToItem(section);
	}
	
	n_2013.onclick = function()
	{
		section = document.getElementById('2013');
		//scrollTo(0, section.offsetTop + 150);
		scrollToItem(section);
	}
	
	n_others.onclick = function()
	{
		//section = document.getElementById('');
		//scrollTo(0, section.offsetTop);
	}
};
