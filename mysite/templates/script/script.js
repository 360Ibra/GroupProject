function IsEnter()
{
	var TextBox_val = document.getElementById("InputBox").value;

	if (TextBox_val != "")
	{
		if (event.keyCode == 13)
			main();
	}
}

function main()
{
	const TextBox_val = document.getElementById("InputBox").value;
	const ResponseText_val = document.getElementById("ResponseText");

	const FormatInput = TextBox_val.toLowerCase().trim();

	if (FormatInput.includes("hi") || FormatInput.includes("hello"))
		ResponseText_val.innerHTML = "Hello!";

	else if (FormatInput.includes("how are you"))
		ResponseText_val.innerHTML = "I'm fine! Thanks for asking!";

	else if (FormatInput.includes("help"))
	ResponseText_val.innerHTML = "What can I help you with?";

	else if (FormatInput.includes("about"))
	ResponseText_val.innerHTML = "This is a web app for predicting stocks and crypto prices";

	else if (FormatInput.includes("open dashboard")) {
		ResponseText_val.innerHTML = "Opening Dashboard...";
		window.open("sign_in.html", "_blank");
	}

	else if (FormatInput.includes("open home")) {
		ResponseText_val.innerHTML = "Opening Home...";
		window.open("index.html", "_blank");
	}

	else
		ResponseText_val.innerHTML = "Sorry, please enter another prompt";
}
