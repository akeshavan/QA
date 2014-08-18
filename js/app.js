var myApp = angular.module('myApp', ['aknavbar']);

myApp.controller("SidebarCtrl", function($scope){
	
	$scope.options = {}
	$scope.images = []
	
	d3.csv("data/freesurferqa.csv", function(error,data) {
        console.log("loaded data")
		data = data
		headers = Object.keys(data[0]);
		$scope.data = data
		headers.shift() //get rid of that empty row name
		headers.shift() //get rid of "image" since its first...change this??
		$scope.headers = headers
		//$scope.headers_rev = headers.reverse();
		$scope.headers.forEach(function(val){$scope.options[val] = []})
		//console.log($scope.headers)
		
		$scope.get_unique = function(header){
			var vals = []
			$scope.data.forEach(function(row){
				vals.push(row[header])
			})
			//angular.element(header).selectpicker()
			uq = _.sortBy(_.uniq(vals))
			//$scope.foo[header] = []
			return uq
		}
		
		$scope.collect = function(){
			//console.log(options)
			images = {}
			data = $scope.data
			data_tmp = []
			for (i=0; i<$scope.headers.length; i++){
				k = $scope.headers.length
				//console.log($scope.headers[k-i-1])
				H = $scope.headers[k-i-1]
			    //console.log(k)
				//console.log($scope.options[H])
				for (o=0;o<$scope.options[H].length; o++){
					
					O = $scope.options[H][o]
					
					data.forEach(function(row){
						if (row[H]==O){
							data_tmp.push(row)
						}
					})	
				}
				
				data = data_tmp
				data_tmp = []
				
			}
			
			$scope.images = data
			console.log(data)
		}
		
		         
	})
		
})


